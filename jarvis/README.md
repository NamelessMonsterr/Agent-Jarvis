# 🤖 Jarvis — AI Developer Companion

Your personal AI command center for research, dataset discovery, devlog commits, and LinkedIn posts.

---

## Architecture

```
User Browser
    ↓
Gradio UI (served by FastAPI)
    ↓
FastAPI Backend
    ↓
Planner Agent → Agent Registry
    ├── Research Agent
    ├── Dataset Agent
    ├── GitHub Agent
    └── LinkedIn Agent
    ↓
LLM Router (GPT → DeepSeek → Qwen → Grok → Ollama)
    ↓
Memory Service
    ├── L1: Conversation Memory (in-process)
    ├── L2: Knowledge Memory (Supabase + pgvector)
    └── L3: Progress Memory (Supabase)
```

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone <your-repo>
cd jarvis/backend
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add at minimum: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
```

### 3. Set up Supabase database

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run both migration files in order:
   - `db/migrations/001_initial.sql`
   - `db/migrations/002_vector_search.sql`
3. Copy your **Project URL** and **Service Role Key** (not anon key) to `.env`

### 4. Set up GitHub devlog repo (optional but recommended)

1. Create a new GitHub repo (e.g. `your-username/jarvis-devlogs`)
2. Generate a Personal Access Token at [github.com/settings/tokens](https://github.com/settings/tokens)
   - Scopes: `repo` (full access)
3. Add `GITHUB_TOKEN` and `GITHUB_REPO` to `.env`

### 5. Run Jarvis

```bash
python main.py
```

Open [http://localhost:8000](http://localhost:8000) — you'll see the Jarvis chat interface.

---

## What Jarvis Can Do

| Say this... | Jarvis does this |
|-------------|-----------------|
| *"Find papers on RAG hallucination"* | Research Agent: summarizes papers, stores insights |
| *"Find datasets for that topic"* | Dataset Agent: searches HuggingFace/Kaggle, uses prior research as context |
| *"Record today: built the auth system, fixed 3 bugs"* | GitHub Agent: creates devlog, commits to your repo |
| *"Generate a LinkedIn post"* | LinkedIn Agent: drafts post from your recent devlogs |
| *"What's the difference between LoRA and full fine-tuning?"* | Direct LLM response (no agent) |

---

## Deployment

### Deploy to Railway

1. Push code to GitHub
2. Create new project at [railway.app](https://railway.app)
3. Connect your GitHub repo
4. Add all environment variables from `.env.example`
5. Railway auto-detects the `Dockerfile` and deploys

### Deploy Frontend (Gradio) — same server

For MVP, the Gradio UI is served directly by the FastAPI backend. No separate frontend deployment needed.

### Migrate to React (post-MVP)

When you're ready for a custom UI:
1. Keep the FastAPI backend as-is
2. Create a React app that calls `/api/chat`
3. Deploy React to Vercel, point `NEXT_PUBLIC_API_URL` to Railway

---

## LLM Fallback Chain

Jarvis tries models in priority order until one succeeds:

1. **GPT-4o-mini** (OpenAI) — default
2. **DeepSeek Chat** — fallback if GPT rate-limited
3. **Qwen Turbo** — fallback #2
4. **Grok Beta** — fallback #3
5. **Gemma 7B (local Ollama)** — offline last resort

You need at least ONE API key configured. Add all of them for maximum resilience.

---

## Project Structure

```
backend/
├── main.py               # FastAPI app + Gradio UI
├── requirements.txt
├── Dockerfile
├── .env.example
├── core/
│   └── config.py         # All settings via environment variables
├── agents/
│   ├── base.py           # BaseAgent — all agents inherit this
│   ├── planner.py        # Routes user intent to the right agent
│   ├── research.py       # Academic research + paper discovery
│   ├── dataset.py        # Dataset discovery (HuggingFace/Kaggle)
│   ├── github.py         # Devlog creation + GitHub commits
│   ├── linkedin.py       # LinkedIn post generation
│   └── registry.py       # Agent name → instance mapping
├── router/
│   └── llm_router.py     # Multi-model fallback router
├── memory/
│   ├── models.py         # Pydantic models (Message, KnowledgeEntry, Devlog)
│   └── service.py        # MemoryService — single interface for all memory ops
├── api/
│   ├── chat.py           # POST /api/chat — main chat endpoint
│   └── health.py         # GET /health, GET /status
└── db/
    ├── client.py         # Supabase client singleton
    └── migrations/
        ├── 001_initial.sql        # Core tables
        └── 002_vector_search.sql  # pgvector RPC function
```

---

## API Reference

### `POST /api/chat`
```json
{
  "message": "Find papers on RAG",
  "session_id": "optional-uuid",
  "user_id": "optional-user-id"
}
```
Returns:
```json
{
  "response": "## 🔬 Research: RAG...",
  "session_id": "uuid",
  "agent_used": "research",
  "model_used": "gpt-4o-mini",
  "fallback_used": false
}
```

### `GET /status`
Returns current model availability and system configuration status.

### `GET /health`
Returns `{"status": "ok"}` — used by Railway for health checks.

---

## Roadmap

- [x] Multi-agent planner system
- [x] LLM router with 5-model fallback
- [x] 3-layer memory system
- [x] Research agent with vector storage
- [x] Dataset agent with cross-agent context
- [x] GitHub devlog commits
- [x] LinkedIn post generator (draft mode)
- [ ] LinkedIn auto-publish (OAuth)
- [ ] Parallel agent execution
- [ ] React frontend
- [ ] User authentication (Supabase Auth)
- [ ] Streaming responses
