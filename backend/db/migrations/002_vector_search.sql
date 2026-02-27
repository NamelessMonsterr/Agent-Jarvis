-- ============================================================
-- pgvector similarity search RPC function
-- Run AFTER 001_initial.sql in Supabase SQL Editor
-- ============================================================

CREATE OR REPLACE FUNCTION match_knowledge(
  query_embedding VECTOR(1536),
  match_user_id   TEXT,
  match_count     INT DEFAULT 5
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
    1 - (k.embedding <=> query_embedding) AS similarity
  FROM knowledge_entries k
  WHERE k.user_id = match_user_id
  ORDER BY k.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
