-- ============================================================
-- Jarvis v3.1 — Memory Aging
-- Run after 003_v3.sql
-- ============================================================

-- Add importance_score and last_accessed to knowledge_entries
ALTER TABLE knowledge_entries
  ADD COLUMN IF NOT EXISTS importance_score FLOAT DEFAULT 0.5
    CHECK (importance_score BETWEEN 0.0 AND 1.0),
  ADD COLUMN IF NOT EXISTS last_accessed    TIMESTAMPTZ DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS access_count     INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_knowledge_aging
  ON knowledge_entries(user_id, importance_score DESC, last_accessed DESC);

-- Function to record an access (called when an entry is retrieved)
CREATE OR REPLACE FUNCTION touch_knowledge_entry(entry_id UUID)
RETURNS VOID AS $$
BEGIN
  UPDATE knowledge_entries
  SET
    last_accessed = NOW(),
    access_count  = access_count + 1,
    -- Boost importance slightly on each access (capped at 1.0)
    importance_score = LEAST(1.0, importance_score + 0.05)
  WHERE id = entry_id;
END;
$$ LANGUAGE plpgsql;

-- Updated match function: incorporates importance_score into ranking
CREATE OR REPLACE FUNCTION match_knowledge_v3(
  query_embedding  VECTOR(1536),
  match_user_id    TEXT,
  match_count      INT DEFAULT 15,
  filter_type      TEXT DEFAULT NULL,
  filter_topic     TEXT DEFAULT NULL
)
RETURNS TABLE (
  id               UUID,
  user_id          TEXT,
  agent            TEXT,
  title            TEXT,
  content          TEXT,
  summary          TEXT,
  source_url       TEXT,
  tags             TEXT[],
  metadata         JSONB,
  created_at       TIMESTAMPTZ,
  last_accessed    TIMESTAMPTZ,
  importance_score FLOAT,
  similarity       FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    k.id, k.user_id, k.agent, k.title, k.content, k.summary,
    k.source_url, k.tags, k.metadata, k.created_at, k.last_accessed,
    k.importance_score,
    1 - (k.embedding <=> query_embedding) AS similarity
  FROM knowledge_entries k
  WHERE
    k.user_id = match_user_id
    AND (filter_type  IS NULL OR k.metadata->>'type'  = filter_type)
    AND (filter_topic IS NULL OR k.metadata->>'topic' ILIKE '%' || filter_topic || '%')
  ORDER BY k.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
