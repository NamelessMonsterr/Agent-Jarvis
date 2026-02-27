"""
BaseAgent v3 — adds execution tracking, timeout, retry, structured logging.
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional

from core.config import settings
from core.logger import AgentLogger
from core.tracing import TraceContext, TraceSpan
from memory.models import Message, AgentRun
from memory.service import memory_service
from router.llm_router import llm_router, LLMRequest, LLMResponse


class BaseAgent(ABC):
    name: str = "base"
    description: str = ""

    def __init__(self):
        self.router = llm_router
        self.memory = memory_service
        self.logger = AgentLogger(self.name)
        self._actual_model: Optional[str] = None   # tracks real model from last LLM call

    @abstractmethod
    async def run(self, user_input: str, session_id: str, user_id: str, **kwargs) -> str:
        ...

    async def execute(
        self,
        user_input: str,
        session_id: str,
        user_id: str,
        trace: Optional[TraceContext] = None,
        **kwargs,
    ) -> str:
        """
        Wrapper around run() that adds:
        - Execution tracking (agent_runs table)
        - Timeout enforcement
        - One automatic retry on failure
        - Execution tracing (spans)
        """
        run = AgentRun(
            session_id=session_id,
            user_id=user_id,
            agent_name=self.name,
            intent=user_input[:500],
        )
        run_id = self.memory.start_agent_run(run)
        start_time = time.time()
        last_error = None

        # Start trace span if tracing is active
        span = trace.start_span(self.name) if trace else None

        for attempt in range(settings.AGENT_MAX_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    self.run(user_input, session_id, user_id, **kwargs),
                    timeout=settings.AGENT_TIMEOUT_S,
                )
                duration_ms = int((time.time() - start_time) * 1000)
                model = self._actual_model or self._last_model()
                self.memory.complete_agent_run(
                    run_id, "success",
                    model=model,
                    duration_ms=duration_ms,
                )
                if span:
                    span.finish("success", model=model)
                self.logger.success(start_time, model or "unknown", f"duration={duration_ms}ms")
                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {settings.AGENT_TIMEOUT_S}s"
                self.logger.warn(f"Attempt {attempt + 1} timed out")
                if attempt < settings.AGENT_MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                self.memory.complete_agent_run(run_id, "timeout", error=last_error)
                if span:
                    span.finish("timeout", error=last_error)
                return f"⚠️ **{self.name} agent timed out.** Please try again."

            except Exception as e:
                last_error = str(e)
                self.logger.failure(last_error, f"attempt={attempt + 1}")
                if attempt < settings.AGENT_MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                duration_ms = int((time.time() - start_time) * 1000)
                self.memory.complete_agent_run(
                    run_id, "failed", error=last_error[:500], duration_ms=duration_ms
                )
                if span:
                    span.finish("failed", error=last_error[:200])
                return f"⚠️ **{self.name} agent failed:** {last_error[:200]}"

        if span:
            span.finish("failed", error="max retries exceeded")
        return f"⚠️ **{self.name} agent failed** after {settings.AGENT_MAX_RETRIES + 1} attempts."

    async def llm(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1500,
        temperature: float = 0.7,
        json_mode: bool = False,
        task_type: str = None,   # "reasoning" | "coding" | "classification" | "creative"
    ) -> LLMResponse:
        response = await self.router.complete(LLMRequest(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            task_type=task_type,
        ))
        self._actual_model = response.model_used    # track real model used
        return response

    def _save_agent_message(self, session_id: str, content: str) -> None:
        self.memory.append_message(
            session_id,
            Message(role="agent", content=content, agent_name=self.name)
        )

    def _get_context_str(self, session_id: str) -> str:
        messages = self.memory.get_context(session_id)
        if not messages:
            return "No prior conversation."
        lines = []
        for m in messages:
            prefix = m.agent_name or m.role
            lines.append(f"[{prefix}]: {m.content[:500]}")
        return "\n".join(lines)

    def _last_model(self) -> Optional[str]:
        if self._actual_model:
            return self._actual_model
        available = self.router.get_available_models()
        return available[0].name if available else None
