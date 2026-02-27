"""
LLM Router — pure orchestration layer.
Knows nothing about HTTP or provider-specific schemas.
All provider details live in router/providers/.
"""
import asyncio
import httpx
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from core.config import settings
from core.logger import router_log
from router.providers.base import BaseProvider, ProviderRequest
from router.providers.openai_compatible import OpenAICompatibleProvider
from router.providers.ollama import OllamaProvider


class ModelState(Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    FAILED = "failed"


class ErrorType(Enum):
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH_FAILURE = "auth_failure"
    PROVIDER_DOWN = "provider_down"
    TIMEOUT = "timeout"
    BAD_RESPONSE = "bad_response"


@dataclass
class ModelConfig:
    name: str
    provider_name: str
    priority: int
    provider: BaseProvider
    timeout_s: int = 15
    state: ModelState = ModelState.AVAILABLE
    degraded_until: Optional[datetime] = None
    recent_timeouts: int = 0
    total_calls: int = 0
    total_failures: int = 0

    def is_available(self) -> bool:
        if self.state == ModelState.FAILED:
            return False
        if self.state == ModelState.DEGRADED:
            if datetime.utcnow() >= self.degraded_until:
                self.state = ModelState.AVAILABLE
                self.recent_timeouts = 0
                return True
            return False
        return True

    def set_degraded(self, ttl_seconds: int) -> None:
        self.state = ModelState.DEGRADED
        self.degraded_until = datetime.utcnow() + timedelta(seconds=ttl_seconds)

    def record_success(self) -> None:
        self.total_calls += 1
        self.recent_timeouts = 0

    def record_failure(self) -> None:
        self.total_calls += 1
        self.total_failures += 1


# ─── Task types for model routing ────────────────────────────────────────────
# Agents pass task_type to get the best model for the job.
# reasoning   → complex multi-step thinking (synthesis, analysis)
# coding       → code generation and debugging
# classification → fast routing/classification calls
# creative     → writing, LinkedIn posts, summaries
# (None)       → use default priority order
TASK_ROUTING: dict[str, list[str]] = {
    "reasoning":      ["gpt-4o-mini", "deepseek-chat", "grok-beta", "qwen-turbo"],
    "coding":         ["deepseek-chat", "gpt-4o-mini", "qwen-turbo", "grok-beta"],
    "classification": ["qwen-turbo", "gpt-4o-mini", "deepseek-chat", "grok-beta"],
    "creative":       ["gpt-4o-mini", "grok-beta", "deepseek-chat", "qwen-turbo"],
}


@dataclass
class LLMRequest:
    system: str
    messages: list[dict]
    max_tokens: int = 1500
    temperature: float = 0.7
    json_mode: bool = False
    # Optional task hint. When set, the router reorders candidates to prefer
    # the best model for this task. Fallback chain is unchanged if unavailable.
    task_type: Optional[str] = None


@dataclass
class LLMResponse:
    content: str
    model_used: str
    provider: str
    fallback_used: bool
    fallback_count: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    finish_reason: str
    errors_encountered: list[str] = field(default_factory=list)


class RouterStats:
    """
    In-process counters for the router's lifetime.
    Readable via GET /status — no DB query needed.
    Resets on process restart (that's fine — Supabase has the durable record).
    """
    def __init__(self):
        self.total_requests: int = 0
        self.total_fallbacks: int = 0
        self.total_failures: int = 0
        self._latency_sum_ms: int = 0
        self._latency_count: int = 0
        self.model_usage: dict[str, int] = {}   # model_name → success count

    def record(self, response: "LLMResponse") -> None:
        self.total_requests += 1
        if response.fallback_used:
            self.total_fallbacks += 1
        self._latency_sum_ms += response.latency_ms
        self._latency_count += 1
        self.model_usage[response.model_used] = (
            self.model_usage.get(response.model_used, 0) + 1
        )

    def record_failure(self) -> None:
        self.total_requests += 1
        self.total_failures += 1

    @property
    def avg_latency_ms(self) -> int:
        if self._latency_count == 0:
            return 0
        return self._latency_sum_ms // self._latency_count

    @property
    def fallback_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_fallbacks / self.total_requests, 3)

    def to_dict(self) -> dict:
        return {
            "total_requests":  self.total_requests,
            "total_fallbacks": self.total_fallbacks,
            "total_failures":  self.total_failures,
            "fallback_rate":   self.fallback_rate,
            "avg_latency_ms":  self.avg_latency_ms,
            "model_usage":     self.model_usage,
        }


class LLMRouter:
    def __init__(self):
        self.models: list[ModelConfig] = self._build_registry()
        self.stats = RouterStats()

    def _build_registry(self) -> list[ModelConfig]:
        registry = []

        if settings.OPENAI_API_KEY:
            registry.append(ModelConfig(
                name="gpt-4o-mini", provider_name="openai", priority=1,
                provider=OpenAICompatibleProvider(
                    api_url="https://api.openai.com/v1/chat/completions",
                    api_key=settings.OPENAI_API_KEY, timeout_s=10),
                timeout_s=10,
            ))

        if settings.DEEPSEEK_API_KEY:
            registry.append(ModelConfig(
                name="deepseek-chat", provider_name="deepseek", priority=2,
                provider=OpenAICompatibleProvider(
                    api_url="https://api.deepseek.com/v1/chat/completions",
                    api_key=settings.DEEPSEEK_API_KEY, timeout_s=15),
                timeout_s=15,
            ))

        if settings.QWEN_API_KEY:
            registry.append(ModelConfig(
                name="qwen-turbo", provider_name="qwen", priority=3,
                provider=OpenAICompatibleProvider(
                    api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                    api_key=settings.QWEN_API_KEY, timeout_s=15),
                timeout_s=15,
            ))

        if settings.GROK_API_KEY:
            registry.append(ModelConfig(
                name="grok-beta", provider_name="xai", priority=4,
                provider=OpenAICompatibleProvider(
                    api_url="https://api.x.ai/v1/chat/completions",
                    api_key=settings.GROK_API_KEY, timeout_s=20),
                timeout_s=20,
            ))

        registry.append(ModelConfig(
            name="gemma:7b", provider_name="ollama", priority=5,
            provider=OllamaProvider(timeout_s=30), timeout_s=30,
        ))

        return sorted(registry, key=lambda m: m.priority)

    def get_available_models(self, task_type: Optional[str] = None) -> list[ModelConfig]:
        available = [m for m in self.models if m.is_available()]
        if not task_type or task_type not in TASK_ROUTING:
            return available  # default priority order

        preferred_order = TASK_ROUTING[task_type]
        # Build a lookup: model name → position in preferred list
        order_map = {name: i for i, name in enumerate(preferred_order)}
        # Models in preferred list come first (sorted by preference),
        # remaining models keep their original priority order at the end
        in_preferred  = sorted(
            [m for m in available if m.name in order_map],
            key=lambda m: order_map[m.name]
        )
        not_preferred = [m for m in available if m.name not in order_map]
        reordered = in_preferred + not_preferred
        if reordered:
            router_log.info(
                f"Task routing [{task_type}]: "
                f"{' → '.join(m.name for m in reordered[:3])}"
            )
        return reordered

    async def complete(self, request: LLMRequest) -> LLMResponse:

        candidates = self.get_available_models(request.task_type)
        if not candidates:
            raise RuntimeError("No LLM models available. Add at least one API key to .env")

        start_time = time.time()
        errors_encountered = []
        fallback_count = 0
        primary_model = candidates[0].name

        for model in candidates:
            # Log every attempt so developers can trace the exact fallback path
            if model.name != primary_model:
                prev_error = errors_encountered[-1] if errors_encountered else "unknown"
                router_log.warning(
                    f"Fallback → {model.name} "
                    f"(#{fallback_count + 1} after {prev_error})"
                )
            else:
                router_log.info(
                    f"Trying {model.name} | task={request.task_type or 'default'}"
                )
            provider_req = ProviderRequest(
                model_name=model.name,
                system=request.system,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                json_mode=request.json_mode,
            )

            try:
                provider_resp = await model.provider.call(provider_req)

                if not provider_resp.content:
                    raise ValueError("empty_response")

                model.record_success()
                latency_ms = int((time.time() - start_time) * 1000)
                router_log.info(f"Model selected: {model.name} | fallback={model.name != primary_model} | latency={latency_ms}ms")

                result = LLMResponse(
                    content=provider_resp.content,
                    model_used=model.name,
                    provider=model.provider_name,
                    fallback_used=(model.name != primary_model),
                    fallback_count=fallback_count,
                    input_tokens=provider_resp.input_tokens,
                    output_tokens=provider_resp.output_tokens,
                    latency_ms=latency_ms,
                    finish_reason=provider_resp.finish_reason,
                    errors_encountered=errors_encountered,
                )

                asyncio.create_task(self._log_metrics(result))
                self.stats.record(result)
                return result

            except httpx.TimeoutException:
                model.recent_timeouts += 1
                errors_encountered.append(f"{model.name}:timeout")
                if model.recent_timeouts >= 2:
                    model.set_degraded(120)
                    router_log.warning(f"{model.name} → DEGRADED (repeated timeouts, 120s)")
                else:
                    router_log.warning(f"{model.name} timed out (timeout #{model.recent_timeouts})")
                fallback_count += 1

            except Exception as e:
                error_str = str(e).lower()
                error_type = self._classify_error(error_str)
                errors_encountered.append(f"{model.name}:{error_type.value}")
                model.record_failure()

                if error_type == ErrorType.AUTH_FAILURE:
                    model.state = ModelState.FAILED
                    router_log.error(f"{model.name} → FAILED permanently (auth error — check API key)")
                elif error_type == ErrorType.QUOTA_EXHAUSTED:
                    model.set_degraded(60)
                    router_log.warning(f"{model.name} → DEGRADED (quota exhausted, 60s)")
                elif error_type == ErrorType.PROVIDER_DOWN:
                    model.set_degraded(120)
                    router_log.warning(f"{model.name} → DEGRADED (provider down, 120s)")
                else:
                    router_log.warning(f"{model.name} failed: {error_type.value} | {str(e)[:80]}")

                fallback_count += 1

        self.stats.record_failure()
        raise RuntimeError(
            f"All LLM models failed. Errors: {errors_encountered}. "
            "Retry in 60 seconds or check your API keys."
        )

    def _classify_error(self, error_str: str) -> ErrorType:
        if any(k in error_str for k in ["rate_limit", "quota", "429"]):
            return ErrorType.QUOTA_EXHAUSTED
        if any(k in error_str for k in ["api_key", "auth", "401", "403"]):
            return ErrorType.AUTH_FAILURE
        if any(k in error_str for k in ["server_error", "500", "503", "502"]):
            return ErrorType.PROVIDER_DOWN
        if "timeout" in error_str:
            return ErrorType.TIMEOUT
        return ErrorType.BAD_RESPONSE

    async def _log_metrics(self, response: LLMResponse) -> None:
        try:
            from db.client import get_supabase
            db = get_supabase()
            db.table("router_metrics").insert({
                "models_attempted": response.errors_encountered or [],
                "model_succeeded": response.model_used,
                "fallback_used": response.fallback_used,
                "fallback_count": response.fallback_count,
                "error_types": response.errors_encountered,
                "total_latency_ms": response.latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }).execute()
        except Exception:
            pass

    @staticmethod
    def extract_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"(\{.*\})", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not extract JSON from: {raw[:200]}")


# Singleton
llm_router = LLMRouter()
