-- ============================================================
-- Jarvis v3 — Database Migration
-- Run AFTER 001_initial.sql and 002_vector_search.sql
-- ============================================================

-- ─── Add metadata + content_hash + chunk support to knowledge_entries ────────
ALTER TABLE knowledge_entries
  ADD COLUMN IF NOT EXISTS metadata    JSONB DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS content_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_knowledge_hash
  ON knowledge_entries(user_id, content_hash)
  WHERE content_hash IS NOT NULL;

-- ─── Add commit_metadata to devlogs ──────────────────────────────────────────
ALTER TABLE devlogs
  ADD COLUMN IF NOT EXISTS commit_metadata JSONB DEFAULT '[]';

-- ─── LinkedIn Posts ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS linkedin_posts (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id            TEXT NOT NULL,
  content            TEXT NOT NULL,
  hook               TEXT DEFAULT '',
  hashtags           TEXT[] DEFAULT '{}',
  word_count         INTEGER DEFAULT 0,
  status             TEXT NOT NULL DEFAULT 'draft'
                     CHECK (status IN ('draft', 'approved', 'posted')),
  source_devlog_ids  TEXT[] DEFAULT '{}',
  tone_profile       JSONB DEFAULT '{}',
  created_at         TIMESTAMPTZ DEFAULT NOW(),
  approved_at        TIMESTAMPTZ,
  posted_at          TIMESTAMPTZ
);

CREATE INDEX idx_linkedin_user_status ON linkedin_posts(user_id, status);

-- ─── Tone Profiles ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tone_profiles (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id        TEXT NOT NULL UNIQUE,
  tone           TEXT DEFAULT 'builder',
  length         TEXT DEFAULT 'medium',
  audience       TEXT DEFAULT 'tech',
  avoid_phrases  TEXT[] DEFAULT '{}',
  updated_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Agent Execution Tracking ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_runs (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id   TEXT NOT NULL,
  user_id      TEXT NOT NULL,
  agent_name   TEXT NOT NULL,
  intent       TEXT,
  status       TEXT NOT NULL DEFAULT 'running'
               CHECK (status IN ('running', 'success', 'failed', 'timeout')),
  model_used   TEXT,
  error        TEXT,
  duration_ms  INTEGER,
  started_at   TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

CREATE INDEX idx_agent_runs_session ON agent_runs(session_id);
CREATE INDEX idx_agent_runs_status  ON agent_runs(status, started_at DESC);

-- ─── Updated vector search function (v3 — supports metadata filtering) ───────
CREATE OR REPLACE FUNCTION match_knowledge_v3(
  query_embedding  VECTOR(1536),
  match_user_id    TEXT,
  match_count      INT DEFAULT 15,
  filter_type      TEXT DEFAULT NULL,
  filter_topic     TEXT DEFAULT NULL
)
RETURNS TABLE (
  id          UUID,
  user_id     TEXT,
  agent       TEXT,
  title       TEXT,
  content     TEXT,
  summary     TEXT,
  source_url  TEXT,
  tags        TEXT[],
  metadata    JSONB,
  created_at  TIMESTAMPTZ,
  similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    k.id,
    k.user_id,
    k.agent,
    k.title,
    k.content,
    k.summary,
    k.source_url,
    k.tags,
    k.metadata,
    k.created_at,
    1 - (k.embedding <=> query_embedding) AS similarity
  FROM knowledge_entries k
  WHERE
    k.user_id = match_user_id
    AND (filter_type IS NULL OR k.metadata->>'type' = filter_type)
    AND (filter_topic IS NULL OR k.metadata->>'topic' ILIKE '%' || filter_topic || '%')
  ORDER BY k.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
