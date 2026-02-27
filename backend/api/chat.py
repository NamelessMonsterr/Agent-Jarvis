"""
Chat API — all responses wrapped in JarvisResponse envelope.
Every endpoint returns the same shape. Frontend never parses ad-hoc dicts.
"""
import time
import uuid
from collections import defaultdict, deque
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional

from agents.planner import planner_agent
from agents.registry import get_agent
from core.auth import get_current_user
from core.config import settings
from core.evaluator import evaluator
from core.exceptions import RateLimitError
from core.logger import api_log
from core.safety import (
    check_injection, InjectionDetected, ConfirmationRequired,
    is_confirmation_message, confirm_action, _pending_confirmations,
)
from core.tracing import TraceContext
from memory.models import Message
from memory.service import memory_service, sanitize_for_llm
from router.llm_router import llm_router, LLMRequest
from schemas.api import (
    JarvisResponse, ChatRequest, ChatData,
    SessionHistoryData, ToneRequest, MemoryCompressData,
)

router = APIRouter(prefix="/api", tags=["chat"])

# ─── Rate limiter (sliding window, per session) ──────────────────────────────
_rate_buckets: dict[str, deque] = defaultdict(
    lambda: deque(maxlen=settings.RATE_LIMIT_PER_MINUTE)
)


def _check_rate_limit(session_id: str) -> None:
    """Raise RateLimitError if session exceeds allowed request rate."""
    now = time.time()
    bucket = _rate_buckets[session_id]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= settings.RATE_LIMIT_PER_MINUTE:
        raise RateLimitError(
            f"Rate limit: {settings.RATE_LIMIT_PER_MINUTE} requests/minute per session."
        )
    bucket.append(now)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/chat", response_model=JarvisResponse)
async def chat(req: ChatRequest, user_id: str = Depends(get_current_user)):
    request_start = time.time()
    request_id = str(uuid.uuid4())[:8]
    session_id = req.session_id or str(uuid.uuid4())
    # user_id now comes from verified JWT, not client-supplied body field

    try:
        _check_rate_limit(session_id)
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))

    clean_message = sanitize_for_llm(req.message)
    if clean_message != req.message:
        api_log.warning(f"[{request_id}] Sanitized potential secret in session {session_id[:8]}")

    # Layer 1: Injection detection — before any LLM sees the input
    try:
        check_injection(clean_message)
    except InjectionDetected as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Layer 2: Confirmation flow — check if user is confirming a pending action
    confirmation_keyword = is_confirmation_message(clean_message)
    if confirmation_keyword and session_id in _pending_confirmations:
        pending_action = _pending_confirmations[session_id]
        confirm_action(session_id, pending_action)
        api_log.info(f"[{request_id}] Action '{pending_action}' confirmed by user")
        # Re-execute with the original intent (confirmation accepted)
        return JarvisResponse.success(
            data=ChatData(
                response=f"✅ **Action confirmed:** `{pending_action}`. Re-running...",
                session_id=session_id,
                agent_used="system",
                model_used="none",
                fallback_used=False,
            ),
            session_id=session_id,
            agent_used="system",
        )

    memory_service.get_or_create_session(session_id, user_id)
    memory_service.append_message(session_id, Message(role="user", content=req.message))

    # Create trace context for this request
    trace = TraceContext(session_id=session_id, request_id=request_id)

    try:
        # Trace the planner step
        planner_span = trace.start_span("planner")
        agent_name, intent = await planner_agent.run(clean_message, session_id, user_id)
        planner_span.finish("success")

        if agent_name == "chat":
            chat_span = trace.start_span("chat")
            response_text, model_used, fallback = await _handle_chat(clean_message, session_id)
            chat_span.finish("success", model=model_used)
        else:
            agent = get_agent(agent_name)
            if not agent:
                response_text = f"Agent '{agent_name}' is not registered."
                model_used, fallback = "none", False
            else:
                response_text = await agent.execute(intent, session_id, user_id, trace=trace)
                model_used = agent._actual_model or "unknown"
                fallback = False

        memory_service.append_message(session_id, Message(role="assistant", content=response_text))

        latency = int((time.time() - request_start) * 1000)

        # Run evaluation (non-blocking heuristics)
        evaluator.evaluate(
            query=clean_message,
            response=response_text,
            agent_name=agent_name,
            model_used=model_used,
            duration_ms=latency,
            session_id=session_id,
            user_id=user_id,
        )

        api_log.info(
            f"[{request_id}] session={session_id[:8]} agent={agent_name} "
            f"model={model_used} latency={latency}ms"
        )
        api_log.debug(trace.to_timeline_str())

        return JarvisResponse.success(
            data=ChatData(
                response=response_text,
                session_id=session_id,
                agent_used=agent_name,
                model_used=model_used,
                fallback_used=fallback,
            ),
            session_id=session_id,
            agent_used=agent_name,
            model_used=model_used,
            fallback_used=fallback,
            latency_ms=latency,
            request_id=request_id,
            trace=trace.to_dict() if settings.DEBUG else None,
        )

    except ConfirmationRequired as e:
        # Destructive action needs user confirmation — surface the prompt
        return JarvisResponse.success(
            data=ChatData(
                response=(
                    f"⚠️ **Confirmation required:** {e.description}\n\n"
                    "Reply **yes** or **confirm** to proceed, or anything else to cancel."
                ),
                session_id=session_id,
                agent_used="system",
                model_used="none",
                fallback_used=False,
            ),
            session_id=session_id,
            agent_used="system",
        )
    except RuntimeError as e:
        api_log.error(f"[{request_id}] Router exhausted in session {session_id[:8]}: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        api_log.error(f"[{request_id}] Unhandled error in session {session_id[:8]}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def _handle_chat(message: str, session_id: str) -> tuple[str, str, bool]:
    context = memory_service.get_context(session_id)
    messages = [m.to_llm_dict() for m in context[:-1]]
    messages.append({"role": "user", "content": message})

    response = await llm_router.complete(LLMRequest(
        system=(
            "You are Jarvis, an AI developer companion. "
            "Be concise and technical. "
            "Help with research, datasets, devlogs, and LinkedIn posts. "
            "Remind users they can say: 'find papers on X', 'find datasets for Y', "
            "'record today: ...', or 'generate LinkedIn post'."
        ),
        messages=messages,
        max_tokens=600,
        temperature=0.7,
    ))
    return response.content, response.model_used, response.fallback_used


@router.get("/session/{session_id}/history", response_model=JarvisResponse)
async def get_history(session_id: str, user_id: str = Depends(get_current_user)):
    messages = memory_service.get_context(session_id, n=50)
    return JarvisResponse.success(
        data=SessionHistoryData(
            session_id=session_id,
            message_count=len(messages),
            messages=[m.model_dump(mode="json") for m in messages],
        ),
        session_id=session_id,
    )


@router.delete("/session/{session_id}", response_model=JarvisResponse)
async def close_session(session_id: str):
    await memory_service.close_session(session_id)
    api_log.info(f"Session closed: {session_id[:8]}")
    return JarvisResponse.success(
        data={"status": "closed", "session_id": session_id},
        session_id=session_id,
    )


@router.post("/tone", response_model=JarvisResponse)
async def set_tone(req: ToneRequest, user_id: str = Depends(get_current_user)):
    from memory.models import ToneProfile
    profile = ToneProfile(**req.model_dump())
    memory_service.save_tone_profile(user_id, profile)
    api_log.info(f"Tone profile saved for user {user_id[:8]}: {profile.tone}/{profile.length}")
    return JarvisResponse.success(data={"status": "saved", "profile": profile.model_dump()})


@router.post("/memory/compress", response_model=JarvisResponse)
async def compress_memory(user_id: str = Depends(get_current_user), older_than_days: int = 30):
    compressed = await memory_service.compress_old_memories(user_id, older_than_days)
    api_log.info(f"Memory compression: {compressed} entries for user {user_id[:8]}")
    return JarvisResponse.success(
        data=MemoryCompressData(compressed=compressed, user_id=user_id)
    )
