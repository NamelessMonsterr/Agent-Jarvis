"""
Jarvis — Main application entry point.
Mounts FastAPI backend + Gradio chat UI on the same server.
"""
import uuid
import uvicorn
import httpx
import gradio as gr
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth import router as auth_router
from api.chat import router as chat_router
from api.health import router as health_router
from core.config import settings, validate_config
from core.logger import get_logger

log = get_logger("main")


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — runs validation at startup, cleanup at shutdown."""
    # Startup
    warnings = validate_config()
    for w in warnings:
        log.warning(f"CONFIG: {w}")
    log.info(f"Jarvis v3.4.0 started — debug={settings.DEBUG} auth={settings.AUTH_REQUIRED}")
    yield
    # Shutdown
    log.info("Jarvis shutting down")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Jarvis API",
    description="AI Developer Companion — Backend API",
    version="3.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)


# ─── Gradio Chat UI (MVP) ────────────────────────────────────────────────────


import os

async def jarvis_chat(message: str, history: list, session_id: str) -> tuple:
    """Gradio chat handler — calls Jarvis backend."""
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        port = os.getenv("PORT", "8000")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/api/chat",
                json={"message": message, "session_id": session_id},
            )
            data = resp.json()
            response = data.get("response", "Error: no response")
            agent = data.get("agent_used", "unknown")
            model = data.get("model_used", "unknown")
            fallback = data.get("fallback_used", False)

        # Add model/agent info as subtle footer
        footer = f"\n\n---\n*🤖 {agent} · {model}{'  ⚡ fallback' if fallback else ''}*"
        full_response = response + footer

    except Exception as e:
        full_response = f"⚠️ Error connecting to Jarvis backend: {str(e)}"

    return full_response, session_id


EXAMPLES = [
    "Summarize the key ideas behind RAG (Retrieval-Augmented Generation)",
    "Find datasets for training a RAG hallucination detection model",
    "Record today's progress: built the LLM router with fallback to DeepSeek and Qwen",
    "Generate a LinkedIn post from my recent work",
    "What agents do you have available?",
]

with gr.Blocks(
    title="Jarvis — AI Developer Companion",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .gradio-container { max-width: 900px; margin: auto; }
    .message.bot { background: #EFF6FF !important; }
    footer { display: none !important; }
    """
) as demo:
    gr.Markdown("""
    # 🤖 Jarvis — AI Developer Companion
    Your AI command center for research, datasets, devlogs, and LinkedIn posts.
    """)

    session_id = gr.State("")

    chatbot = gr.Chatbot(
        label="Jarvis",
        height=520,
        show_copy_button=True,
        render_markdown=True,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask Jarvis anything... research a topic, find datasets, log your progress",
            show_label=False,
            scale=9,
            autofocus=True,
        )
        submit = gr.Button("Send", variant="primary", scale=1)

    gr.Examples(
        examples=EXAMPLES,
        inputs=msg,
        label="Try these",
    )

    with gr.Accordion("ℹ️ What can Jarvis do?", open=False):
        gr.Markdown("""
        | Command | Example |
        |---------|---------|
        | 🔬 Research | *"Find papers on transformer attention mechanisms"* |
        | 📊 Datasets | *"Find datasets for sentiment analysis"* |
        | 📝 Devlog | *"Record today: fixed authentication bug, wrote tests"* |
        | 💼 LinkedIn | *"Generate a LinkedIn post about my week"* |
        | 💬 General | *"What's the difference between RAG and fine-tuning?"* |
        """)

    async def respond(message, history, sid):
        history = history or []
        response, new_sid = await jarvis_chat(message, history, sid)
        history.append((message, response))
        return "", history, new_sid

    msg.submit(respond, [msg, chatbot, session_id], [msg, chatbot, session_id])
    submit.click(respond, [msg, chatbot, session_id], [msg, chatbot, session_id])


# ─── Mount Gradio on FastAPI ──────────────────────────────────────────────────

app = gr.mount_gradio_app(app, demo, path="/")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
