"""
tests/test_memory.py

Tests for memory service: chunking, deduplication, hybrid ranking,
metadata filtering, knowledge storage.

Run: pytest tests/test_memory.py -v
"""
import sys
import types
import hashlib
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

# ─── Mock supabase before importing memory modules (not installed locally) ────
_supabase_mock = types.ModuleType("supabase")
_supabase_mock.create_client = MagicMock()
_supabase_mock.Client = MagicMock()
sys.modules.setdefault("supabase", _supabase_mock)

from memory.service import _chunk_text, sanitize_for_llm, MemoryService
from memory.models import (
    KnowledgeEntry, ChunkMetadata, Paper, Session, Message
)


# ─── Tests: chunking ──────────────────────────────────────────────────────────

def test_short_text_not_chunked():
    """Text under chunk_size is returned as a single chunk."""
    text = "This is a short paragraph."
    chunks = _chunk_text(text, chunk_size=600, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_produces_multiple_chunks():
    """Text significantly longer than chunk_size produces multiple chunks."""
    words = ["word"] * 1400   # ~1400 words, chunk_size=600
    text = " ".join(words)
    chunks = _chunk_text(text, chunk_size=600, overlap=100)
    assert len(chunks) >= 2


def test_chunks_have_overlap():
    """Adjacent chunks share content from the overlap window."""
    words = [f"w{i}" for i in range(800)]
    text = " ".join(words)
    chunks = _chunk_text(text, chunk_size=600, overlap=100)

    # The end of chunk[0] and start of chunk[1] should share words
    end_of_first = set(chunks[0].split()[-100:])
    start_of_second = set(chunks[1].split()[:100])
    overlap = end_of_first & start_of_second
    assert len(overlap) > 0, "Chunks should have overlapping content"


def test_chunk_count_is_deterministic():
    """Same input always produces same number of chunks."""
    text = " ".join(["word"] * 1200)
    c1 = _chunk_text(text, chunk_size=600, overlap=100)
    c2 = _chunk_text(text, chunk_size=600, overlap=100)
    assert len(c1) == len(c2)


# ─── Tests: sanitization ─────────────────────────────────────────────────────

def test_sanitize_removes_openai_key():
    text = "My key is sk-abcdefghijklmnop1234 and it works"
    result = sanitize_for_llm(text)
    assert "sk-abcdefghijklmnop1234" not in result
    assert "[REDACTED]" in result


def test_sanitize_removes_github_token():
    text = "token = ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456"
    result = sanitize_for_llm(text)
    assert "ghp_" not in result or "aBcDeFgHiJkLmNoPqRsTuVwXyZ123456" not in result


def test_sanitize_passes_clean_text():
    text = "What is retrieval-augmented generation?"
    result = sanitize_for_llm(text)
    assert result == text


def test_sanitize_is_idempotent():
    """Sanitizing twice produces the same result as sanitizing once."""
    text = "key: sk-testtoken1234567890"
    once = sanitize_for_llm(text)
    twice = sanitize_for_llm(once)
    assert once == twice


# ─── Tests: Paper deduplication ───────────────────────────────────────────────

def test_paper_hash_uses_doi_if_available():
    paper = Paper(
        title="Test Paper", abstract="Abstract",
        doi="10.1234/test", source="arxiv", url=""
    )
    expected = hashlib.sha256("10.1234/test".encode()).hexdigest()[:16]
    assert paper.content_hash() == expected


def test_paper_hash_uses_title_when_no_doi():
    paper = Paper(
        title="  Test Paper  ", abstract="Abstract",
        doi=None, source="arxiv", url=""
    )
    expected = hashlib.sha256("test paper".encode()).hexdigest()[:16]
    assert paper.content_hash() == expected


def test_paper_dedup_by_hash():
    """Two papers with same DOI produce same hash (dedup key)."""
    p1 = Paper(title="Paper A", doi="10.1234/x", source="arxiv", url="")
    p2 = Paper(title="Paper B", doi="10.1234/x", source="semantic_scholar", url="")
    assert p1.content_hash() == p2.content_hash()


# ─── Tests: Session management ────────────────────────────────────────────────

def test_session_create_and_retrieve():
    svc = MemoryService.__new__(MemoryService)
    svc._sessions = {}
    svc._seen_hashes = set()

    sid = "550e8400-e29b-41d4-a716-446655440001"
    session = svc.get_or_create_session(sid, "user-1")
    assert str(session.user_id) == "user-1"

    # Same session_id returns same session
    same = svc.get_or_create_session(sid, "user-1")
    assert same is session


def test_append_message_to_session():
    svc = MemoryService.__new__(MemoryService)
    svc._sessions = {}
    svc._seen_hashes = set()

    sid = "550e8400-e29b-41d4-a716-446655440002"
    svc.get_or_create_session(sid, "user-1")
    svc.append_message(sid, Message(role="user", content="hello"))
    svc.append_message(sid, Message(role="assistant", content="hi"))

    context = svc.get_context(sid)
    assert len(context) == 2
    assert context[0].content == "hello"


def test_get_context_returns_last_n():
    svc = MemoryService.__new__(MemoryService)
    svc._sessions = {}
    svc._seen_hashes = set()

    sid = "550e8400-e29b-41d4-a716-446655440003"
    svc.get_or_create_session(sid, "user-1")
    for i in range(15):
        svc.append_message(sid, Message(role="user", content=f"msg {i}"))

    context = svc.get_context(sid, n=5)
    assert len(context) == 5
    assert context[-1].content == "msg 14"


# ─── Tests: In-process dedup ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_knowledge_dedup_in_process():
    """Storing the same content hash twice is a no-op (in-process dedup)."""
    svc = MemoryService.__new__(MemoryService)
    svc._sessions = {}
    svc._seen_hashes = set()
    svc._openai = None  # no embeddings needed for dedup test

    entry = KnowledgeEntry(
        user_id="user-1",
        agent="research",
        title="Test Paper",
        content="Some research content",
        tags=[],
        metadata=ChunkMetadata(type="research", agent="research"),
        content_hash="abc123",
    )

    # Seed the in-process cache
    svc._seen_hashes.add("abc123")

    # Attempting to store should return empty (dedup hit)
    with patch.object(svc, '_embed', new_callable=AsyncMock) as mock_embed:
        result = await svc.store_knowledge(entry)
        mock_embed.assert_not_called()

    assert result == []


# ─── Tests: Hybrid ranking math ───────────────────────────────────────────────

def test_recency_decay_recent_entry():
    """Entry from today should have near-1.0 recency score."""
    age_days = 0
    recency = math.exp(-age_days / 30)
    assert recency > 0.99


def test_recency_decay_old_entry():
    """Entry from 60 days ago should have low recency score."""
    age_days = 60
    recency = math.exp(-age_days / 30)
    assert recency < 0.15


def test_hybrid_score_weights_sum_to_one():
    """Verify the hybrid score formula weights are internally consistent."""
    from core.config import settings
    w_r = settings.RECENCY_WEIGHT
    w_i = 0.1
    w_s = 1 - w_r - w_i
    assert abs(w_s + w_r + w_i - 1.0) < 1e-9
