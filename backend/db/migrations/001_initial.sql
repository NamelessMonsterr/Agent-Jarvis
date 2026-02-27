-- ============================================================
-- Jarvis — Initial Database Schema
-- Run this in Supabase SQL Editor
-- ============================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ────────────────────────────────────────────────────────────
-- Sessions (L1 Conversation Memory — persisted on close)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      TEXT NOT NULL,
  messages     JSONB NOT NULL DEFAULT '[]',
  active_agents TEXT[] DEFAULT '{}',
  metadata     JSONB DEFAULT '{}',
  created_at   TIMESTAMPTZ DEFAULT NOW(),
  closed_at    TIMESTAMPTZ
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);

-- ────────────────────────────────────────────────────────────
-- Knowledge Entries (L2 — vector semantic memory)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge_entries (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      TEXT NOT NULL,
  agent        TEXT NOT NULL,  -- 'research' | 'dataset' | 'manual'
  title        TEXT NOT NULL,
  content      TEXT NOT NULL,
  summary      TEXT,
  source_url   TEXT,
  tags         TEXT[] DEFAULT '{}',
  embedding    VECTOR(1536),
  session_id   UUID REFERENCES sessions(id) ON DELETE SET NULL,
  created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_knowledge_user_id ON knowledge_entries(user_id);
CREATE INDEX idx_knowledge_tags ON knowledge_entries USING GIN(tags);
CREATE INDEX idx_knowledge_embedding ON knowledge_entries
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ────────────────────────────────────────────────────────────
-- Devlogs (L3 — progress memory)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS devlogs (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      TEXT NOT NULL,
  date         DATE NOT NULL,
  summary      TEXT NOT NULL,
  tasks        JSONB DEFAULT '[]',   -- [{task, status, notes}]
  commits      TEXT[] DEFAULT '{}',  -- GitHub commit SHAs
  topics       TEXT[] DEFAULT '{}',
  mood         SMALLINT CHECK (mood BETWEEN 1 AND 5),
  github_path  TEXT,
  created_at   TIMESTAMPTZ DEFAULT NOW(),
  updated_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_devlogs_user_date ON devlogs(user_id, date);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER devlogs_updated_at
  BEFORE UPDATE ON devlogs
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ────────────────────────────────────────────────────────────
-- LLM Router Metrics (observability)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS router_metrics (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id       UUID,
  models_attempted TEXT[] NOT NULL,
  model_succeeded  TEXT NOT NULL,
  fallback_used    BOOLEAN DEFAULT FALSE,
  fallback_count   SMALLINT DEFAULT 0,
  error_types      TEXT[] DEFAULT '{}',
  total_latency_ms INTEGER,
  input_tokens     INTEGER,
  output_tokens    INTEGER,
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_metrics_session ON router_metrics(session_id);
CREATE INDEX idx_metrics_created ON router_metrics(created_at);
