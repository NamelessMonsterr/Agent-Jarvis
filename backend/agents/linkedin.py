"""
LinkedIn Agent v3 — DRAFT lifecycle, tone profile, schemas format, post dedup.
"""
import hashlib
from agents.base import BaseAgent
from core.exceptions import AgentError
from memory.models import LinkedInPost, ToneProfile
from memory.service import sanitize_for_llm
from router.llm_router import LLMRouter
from schemas.api import LinkedInPostDraft
from schemas.response import format_linkedin_draft, format_linkedin_no_content
from core.logger import get_logger

log = get_logger(__name__)


def _post_prompt(p: ToneProfile) -> str:
    length = {"short": "100-150", "medium": "150-250", "long": "250-350"}[p.length]
    tone = {
        "builder": "honest builder tone — show the real journey including struggles",
        "technical": "precise, technical — developers are reading this",
        "casual": "relaxed and human, like talking to a peer",
        "professional": "polished thought leadership",
    }.get(p.tone, "builder")
    avoid = f"\nNEVER use: {', '.join(p.avoid_phrases)}" if p.avoid_phrases else ""
    return (
        f"Write a LinkedIn post for a developer building in public.\n"
        f"Tone: {tone}\nLength: {length} words\nAudience: {p.audience}{avoid}\n\n"
        "Rules:\n"
        "- Hook must NOT start with 'I built' or 'Excited to share'\n"
        "- One concrete insight or takeaway\n"
        "- End with one open question\n"
        "- 5-8 hashtags on the last line\n\n"
        "Respond in JSON only:\n"
        '{"post": "...", "hook": "...", "hashtags": ["#tag"], "word_count": <n>}'
    )


class LinkedInAgent(BaseAgent):
    name = "linkedin"
    description = "Generates LinkedIn post drafts from devlogs and research. Never auto-posts."

    async def run(self, user_input: str, session_id: str, user_id: str, **kwargs) -> str:
        start_time = self.logger.start(session_id, user_input)

        tone_profile = self.memory.get_tone_profile(user_id)

        # Content priority: devlogs → research → empty
        devlogs = self.memory.get_recent_devlogs(user_id, days=7)
        research = await self.memory.search_knowledge(user_input, user_id, k=2, filter_type="research")

        if not devlogs and not research:
            out = format_linkedin_no_content()
            self._save_agent_message(session_id, out)
            return out

        context_parts = []
        if devlogs:
            context_parts.append("### Recent Devlogs")
            for log in devlogs[:4]:
                context_parts.append(f"**{log.date}**: {log.summary}")
                if log.topics:
                    context_parts.append(f"Topics: {', '.join(log.topics)}")
        if research:
            context_parts.append("\n### Recent Research")
            for e in research:
                context_parts.append(f"- {e.title}: {e.summary or e.content[:150]}")

        context = "\n".join(context_parts)
        response = await self.llm(
            system=_post_prompt(tone_profile),
            task_type="creative",
            messages=[{"role": "user", "content": f"{context}\n\nUser note: {sanitize_for_llm(user_input)}"}],
            max_tokens=600, temperature=0.85, json_mode=True,
        )

        try:
            raw = LLMRouter.extract_json(response.content)
            draft = LinkedInPostDraft.from_llm(raw)
        except (ValueError, Exception) as e:
            raise AgentError(self.name, f"LLM returned unparseable post structure: {e}")

        post_content = draft.post

        # Dedup: skip if identical post was already drafted recently
        post_hash = hashlib.sha256(post_content.encode()).hexdigest()[:16]
        existing = self.memory.tag_search([f"post_hash:{post_hash}"], user_id)
        if existing:
            self.logger.skip(f"duplicate post hash {post_hash}")
            return f"⚠️ This post was already generated recently. Try: *'generate a different LinkedIn post'*"

        post = LinkedInPost(
            user_id=user_id,
            content=post_content,
            hook=draft.hook,
            hashtags=draft.hashtags,
            word_count=draft.word_count or len(post_content.split()),
            status="draft",
            source_devlog_ids=[str(d.id) for d in devlogs[:4] if d.id],
            tone_profile=tone_profile,
        )
        saved = self.memory.save_linkedin_post(post)

        # Tag for dedup
        from memory.models import KnowledgeEntry, ChunkMetadata
        await self.memory.store_knowledge(KnowledgeEntry(
            user_id=user_id, agent="manual",
            title=f"LinkedIn draft {str(saved.id)[:8]}",
            content=post_content[:200],
            tags=[f"post_hash:{post_hash}"],
            metadata=ChunkMetadata(type="manual", agent="linkedin", topic="linkedin"),
        ))

        self.logger.success(start_time, response.model_used, f"draft id={str(saved.id)[:8]}")
        out = format_linkedin_draft(saved, response.model_used)
        self._save_agent_message(session_id, out)
        return out


linkedin_agent = LinkedInAgent()
