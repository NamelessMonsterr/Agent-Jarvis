"""
MemoryService — the ONLY way agents interact with memory.
v3: Real RAG with chunking, metadata filtering, hybrid ranking, dedup, compression.
"""
import hashlib
import math
from datetime import date, datetime, timedelta
from typing import Optional
from uuid import UUID

from openai import AsyncOpenAI

from core.config import settings
from core.logger import memory_log
from db.client import get_supabase
from memory.models import (
    Message, Session, KnowledgeEntry, ChunkMetadata,
    Devlog, Task, CommitMetadata, AgentRun, LinkedInPost, ToneProfile
)


# ─── Chunking ────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping token-approximate chunks."""
    chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
    overlap = overlap or settings.RAG_CHUNK_OVERLAP
    # Approximate tokens as words (close enough for chunking)
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ─── Prompt Sanitization ─────────────────────────────────────────────────────

_SECRET_PATTERNS = [
    "sk-", "ghp_", "gsk_", "api_key", "token", "password", "secret",
    "SUPABASE_SERVICE_KEY", "GITHUB_TOKEN",
]

def sanitize_for_llm(text: str) -> str:
    """Remove potential secrets before sending text to LLM APIs."""
    for pattern in _SECRET_PATTERNS:
        if pattern.lower() in text.lower():
            # Replace anything that looks like a long token
            import re
            text = re.sub(
                r'(sk-|ghp_|gsk_|eyJ)[A-Za-z0-9_\-\.]{10,}',
                '[REDACTED]',
                text
            )
    return text


# ─── MemoryService ────────────────────────────────────────────────────────────

class MemoryService:
    _MAX_SEEN_HASHES = 10_000   # Bounded dedup cache — evicts oldest when full

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
        self._seen_hashes: set[str] = set()  # in-process dedup cache

    # ─── L1: Conversation Memory ─────────────────────────────────────────────

    def get_or_create_session(self, session_id: str, user_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(id=session_id, user_id=user_id)
        return self._sessions[session_id]

    def get_context(self, session_id: str, n: int = None) -> list[Message]:
        n = n or settings.MAX_CONTEXT_MESSAGES
        session = self._sessions.get(session_id)
        return session.messages[-n:] if session else []

    def append_message(self, session_id: str, message: Message) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.messages.append(message)
            if message.agent_name and message.agent_name not in session.active_agents:
                session.active_agents.append(message.agent_name)

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        session.closed_at = datetime.utcnow()
        try:
            db = get_supabase()
            db.table("sessions").upsert({
                "id": str(session.id),
                "user_id": session.user_id,
                "messages": [m.model_dump(mode="json") for m in session.messages],
                "active_agents": session.active_agents,
                "metadata": session.metadata,
                "closed_at": session.closed_at.isoformat(),
            }).execute()
        except Exception as e:
            memory_log.warning(f"Failed to persist session {session_id}: {e}")
        finally:
            del self._sessions[session_id]

    def cleanup_stale_sessions(self) -> int:
        """Remove sessions older than SESSION_TTL_HOURS. Call periodically."""
        ttl = timedelta(hours=settings.SESSION_TTL_HOURS)
        now = datetime.utcnow()
        stale = [
            sid for sid, s in self._sessions.items()
            if (now - s.created_at) > ttl
        ]
        for sid in stale:
            memory_log.info(f"Evicting stale session {sid[:8]} (age > {settings.SESSION_TTL_HOURS}h)")
            try:
                # Best-effort persist before eviction
                import asyncio
                asyncio.create_task(self.close_session(sid))
            except Exception:
                del self._sessions[sid]
        return len(stale)

    def _add_seen_hash(self, h: str) -> None:
        """Add hash to dedup cache with eviction when full."""
        if len(self._seen_hashes) >= self._MAX_SEEN_HASHES:
            # Evict ~20% of the oldest entries (set has no ordering, just clear a chunk)
            evict_count = self._MAX_SEEN_HASHES // 5
            it = iter(self._seen_hashes)
            to_remove = [next(it) for _ in range(evict_count)]
            self._seen_hashes -= set(to_remove)
            memory_log.debug(f"Evicted {evict_count} entries from dedup cache")
        self._seen_hashes.add(h)

    # ─── L2: Knowledge Memory (RAG) ───────────────────────────────────────────

    async def store_knowledge(self, entry: KnowledgeEntry) -> list[str]:
        """
        Chunk, deduplicate, embed and store a knowledge entry.
        Returns list of stored entry IDs (one per chunk).
        """
        # Dedup check
        content_hash = entry.content_hash or hashlib.sha256(
            entry.content.encode()
        ).hexdigest()[:16]

        if content_hash in self._seen_hashes:
            memory_log.debug(f"Dedup hit (in-process): {entry.title[:50]}")
            return []

        # Check DB dedup
        try:
            db = get_supabase()
            existing = (
                db.table("knowledge_entries")
                .select("id")
                .eq("user_id", entry.user_id)
                .eq("content_hash", content_hash)
                .limit(1)
                .execute()
            )
            if existing.data:
                memory_log.debug(f"Dedup hit (DB): {entry.title[:50]}")
                self._seen_hashes.add(content_hash)
                return []
        except Exception:
            pass  # On DB error, continue to store

        self._seen_hashes.add(content_hash)

        # Chunk content
        chunks = _chunk_text(entry.content)
        stored_ids = []

        for i, chunk in enumerate(chunks):
            embedding = await self._embed(chunk)
            chunk_meta = entry.metadata.model_copy(update={"chunk_index": i})

            row = {
                "user_id": entry.user_id,
                "agent": entry.agent,
                "title": entry.title if len(chunks) == 1 else f"{entry.title} [chunk {i+1}/{len(chunks)}]",
                "content": chunk,
                "summary": entry.summary if i == 0 else None,
                "source_url": entry.source_url,
                "tags": entry.tags,
                "metadata": chunk_meta.model_dump(mode="json"),
                "content_hash": content_hash if i == 0 else None,
                "session_id": str(entry.session_id) if entry.session_id else None,
                "embedding": embedding,
            }
            try:
                result = db.table("knowledge_entries").insert(row).execute()
                stored_ids.append(result.data[0]["id"])
            except Exception as e:
                memory_log.warning(f"Failed to store chunk {i}: {e}")

        memory_log.info(f"Stored '{entry.title[:50]}' as {len(stored_ids)} chunk(s)")
        return stored_ids

    async def search_knowledge(
        self,
        query: str,
        user_id: str,
        k: int = None,
        filter_type: Optional[str] = None,
        filter_topic: Optional[str] = None,
    ) -> list[KnowledgeEntry]:
        """
        Hybrid retrieval: semantic similarity + recency score.
        Optional metadata filters: type (research/dataset/devlog), topic.
        """
        k = k or settings.RAG_TOP_K
        if not self._openai:
            return []

        embedding = await self._embed(query)
        db = get_supabase()

        try:
            # Build RPC params — filtering happens inside the SQL function
            rpc_params = {
                "query_embedding": embedding,
                "match_user_id": user_id,
                "match_count": k * 3,  # over-fetch then re-rank
            }
            if filter_type:
                rpc_params["filter_type"] = filter_type
            if filter_topic:
                rpc_params["filter_topic"] = filter_topic

            result = db.rpc("match_knowledge_v3", rpc_params).execute()
            rows = result.data or []
        except Exception as e:
            memory_log.warning(f"Vector search failed: {e}")
            return []

        # Hybrid re-ranking: semantic similarity + recency decay + importance_score
        now = datetime.utcnow()
        scored = []
        for row in rows:
            similarity = float(row.get("similarity", 0))
            importance = float(row.get("importance_score", 0.5))

            created_at_str = row.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                age_days = (now - created_at.replace(tzinfo=None)).days
                recency = math.exp(-age_days / 30)   # exponential decay over 30 days
            except Exception:
                recency = 0.5

            # Weighted combination: semantic is dominant, recency and importance are adjustments
            w_r = settings.RECENCY_WEIGHT
            w_i = 0.1   # importance weight — small nudge, not dominant
            hybrid = (1 - w_r - w_i) * similarity + w_r * recency + w_i * importance
            scored.append((hybrid, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        entries = []
        touch_ids = []
        for _, row in top:
            meta_raw = row.get("metadata") or {}
            meta = ChunkMetadata(
                type=meta_raw.get("type", "manual"),
                agent=meta_raw.get("agent", "unknown"),
                topic=meta_raw.get("topic", ""),
            )
            entries.append(KnowledgeEntry(
                id=row.get("id"),
                user_id=row.get("user_id", user_id),
                agent=row.get("agent", "manual"),
                title=row.get("title", ""),
                content=row.get("content", ""),
                summary=row.get("summary"),
                source_url=row.get("source_url"),
                tags=row.get("tags", []),
                metadata=meta,
            ))
            if row.get("id"):
                touch_ids.append(row["id"])

        # Fire-and-forget: update last_accessed + access_count for retrieved entries
        if touch_ids:
            try:
                for eid in touch_ids:
                    db.rpc("touch_knowledge_entry", {"entry_id": eid}).execute()
            except Exception:
                pass  # Never let access tracking break retrieval

        return entries

    def tag_search(self, tags: list[str], user_id: str) -> list[KnowledgeEntry]:
        db = get_supabase()
        result = (
            db.table("knowledge_entries")
            .select("*")
            .eq("user_id", user_id)
            .contains("tags", tags)
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        return [self._row_to_knowledge(row) for row in (result.data or [])]

    async def compress_old_memories(self, user_id: str, older_than_days: int = 30) -> int:
        """
        Summarize old memory entries to prevent context explosion.
        Groups entries by topic, summarizes each group, deletes originals.
        Returns number of entries compressed.
        """
        cutoff = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()
        db = get_supabase()

        try:
            old = (
                db.table("knowledge_entries")
                .select("id, title, content, tags, metadata, importance_score, access_count")
                .eq("user_id", user_id)
                .lt("created_at", cutoff)
                .lt("importance_score", 0.7)   # never compress high-importance entries
                .limit(50)
                .execute()
            )
            entries = old.data or []
        except Exception as e:
            memory_log.warning(f"Memory compression query failed: {e}")
            return 0

        if len(entries) < 5:
            return 0

        # Group by topic tag
        groups: dict[str, list] = {}
        for e in entries:
            meta = e.get("metadata") or {}
            topic = meta.get("topic", "general")
            groups.setdefault(topic, []).append(e)

        compressed = 0
        for topic, group_entries in groups.items():
            if len(group_entries) < 3:
                continue
            combined = "\n\n".join(e["content"][:400] for e in group_entries)
            summary = f"[Compressed {len(group_entries)} entries on '{topic}']\n{combined[:800]}"

            new_entry = KnowledgeEntry(
                user_id=user_id,
                agent="manual",
                title=f"Memory summary: {topic}",
                content=summary,
                summary=f"Compressed {len(group_entries)} older entries about {topic}",
                tags=[topic, "compressed"],
                metadata=ChunkMetadata(type="manual", agent="system", topic=topic),
            )
            await self.store_knowledge(new_entry)

            # Delete originals
            ids = [e["id"] for e in group_entries]
            try:
                db.table("knowledge_entries").delete().in_("id", ids).execute()
                compressed += len(ids)
            except Exception as e:
                memory_log.warning(f"Failed to delete compressed entries: {e}")

        memory_log.info(f"Compressed {compressed} memory entries for user {user_id[:8]}")
        return compressed

    # ─── L3: Progress Memory ─────────────────────────────────────────────────

    def upsert_devlog(self, devlog: Devlog) -> Devlog:
        db = get_supabase()
        row = {
            "user_id": devlog.user_id,
            "date": devlog.date.isoformat(),
            "summary": devlog.summary,
            "tasks": [t.model_dump() for t in devlog.tasks],
            "commits": devlog.commits,
            "commit_metadata": [c.model_dump(mode="json") for c in devlog.commit_metadata],
            "topics": devlog.topics,
            "mood": devlog.mood,
            "github_path": devlog.github_path,
        }
        result = db.table("devlogs").upsert(row, on_conflict="user_id,date").execute()
        saved = result.data[0]
        devlog.id = saved["id"]
        return devlog

    def get_devlog(self, user_id: str, log_date: date) -> Optional[Devlog]:
        db = get_supabase()
        result = (
            db.table("devlogs")
            .select("*")
            .eq("user_id", user_id)
            .eq("date", log_date.isoformat())
            .limit(1)
            .execute()
        )
        return self._row_to_devlog(result.data[0]) if result.data else None

    def get_recent_devlogs(self, user_id: str, days: int = 7) -> list[Devlog]:
        db = get_supabase()
        result = (
            db.table("devlogs")
            .select("*")
            .eq("user_id", user_id)
            .order("date", desc=True)
            .limit(days)
            .execute()
        )
        return [self._row_to_devlog(row) for row in (result.data or [])]

    # ─── Agent Run Tracking ───────────────────────────────────────────────────

    def start_agent_run(self, run: AgentRun) -> str:
        try:
            db = get_supabase()
            result = db.table("agent_runs").insert({
                "session_id": run.session_id,
                "user_id": run.user_id,
                "agent_name": run.agent_name,
                "intent": run.intent[:500],
                "status": "running",
                "started_at": run.started_at.isoformat(),
            }).execute()
            return result.data[0]["id"]
        except Exception as e:
            memory_log.debug(f"agent_runs insert failed: {e}")
            return ""

    def complete_agent_run(self, run_id: str, status: str, model: str = None, error: str = None, duration_ms: int = None) -> None:
        if not run_id:
            return
        try:
            db = get_supabase()
            db.table("agent_runs").update({
                "status": status,
                "model_used": model,
                "error": error,
                "duration_ms": duration_ms,
                "completed_at": datetime.utcnow().isoformat(),
            }).eq("id", run_id).execute()
        except Exception as e:
            memory_log.debug(f"agent_runs update failed: {e}")

    # ─── LinkedIn Posts ───────────────────────────────────────────────────────

    def save_linkedin_post(self, post: LinkedInPost) -> LinkedInPost:
        db = get_supabase()
        row = {
            "user_id": post.user_id,
            "content": post.content,
            "hook": post.hook,
            "hashtags": post.hashtags,
            "word_count": post.word_count,
            "status": post.status,
            "source_devlog_ids": post.source_devlog_ids,
            "tone_profile": post.tone_profile.model_dump() if post.tone_profile else None,
        }
        if post.id:
            result = db.table("linkedin_posts").update(row).eq("id", str(post.id)).execute()
        else:
            result = db.table("linkedin_posts").insert(row).execute()
        post.id = result.data[0]["id"]
        return post

    def get_tone_profile(self, user_id: str) -> ToneProfile:
        try:
            db = get_supabase()
            result = (
                db.table("tone_profiles")
                .select("*")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if result.data:
                d = result.data[0]
                return ToneProfile(
                    tone=d.get("tone", "builder"),
                    length=d.get("length", "medium"),
                    audience=d.get("audience", "tech"),
                    avoid_phrases=d.get("avoid_phrases", []),
                )
        except Exception:
            pass
        return ToneProfile()

    def save_tone_profile(self, user_id: str, profile: ToneProfile) -> None:
        try:
            db = get_supabase()
            db.table("tone_profiles").upsert({
                "user_id": user_id,
                "tone": profile.tone,
                "length": profile.length,
                "audience": profile.audience,
                "avoid_phrases": profile.avoid_phrases,
            }, on_conflict="user_id").execute()
        except Exception as e:
            memory_log.warning(f"Failed to save tone profile: {e}")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        if not self._openai:
            return [0.0] * 1536
        try:
            resp = await self._openai.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=text[:8000],
            )
            return resp.data[0].embedding
        except Exception as e:
            memory_log.warning(f"Embedding failed: {e}")
            return [0.0] * 1536

    def _row_to_devlog(self, row: dict) -> Devlog:
        commits_meta_raw = row.get("commit_metadata") or []
        return Devlog(
            id=row.get("id"),
            user_id=row["user_id"],
            date=date.fromisoformat(row["date"]),
            summary=row["summary"],
            tasks=[Task(**t) for t in (row.get("tasks") or [])],
            commits=row.get("commits", []),
            commit_metadata=[CommitMetadata(**c) for c in commits_meta_raw],
            topics=row.get("topics", []),
            mood=row.get("mood"),
            github_path=row.get("github_path"),
        )

    def _row_to_knowledge(self, row: dict) -> KnowledgeEntry:
        meta_raw = row.get("metadata") or {}
        return KnowledgeEntry(
            id=row.get("id"),
            user_id=row.get("user_id", ""),
            agent=row.get("agent", "manual"),
            title=row.get("title", ""),
            content=row.get("content", ""),
            summary=row.get("summary"),
            source_url=row.get("source_url"),
            tags=row.get("tags", []),
            metadata=ChunkMetadata(
                type=meta_raw.get("type", "manual"),
                agent=meta_raw.get("agent", "unknown"),
                topic=meta_raw.get("topic", ""),
            ),
        )


# Singleton
memory_service = MemoryService()
