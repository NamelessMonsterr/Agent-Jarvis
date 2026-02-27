"""
tests/test_router.py

Tests for LLM Router: fallback behavior, model health state,
task-based routing, provider normalization.

Run: pytest tests/test_router.py -v
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from router.llm_router import LLMRouter, LLMRequest, ModelState, TASK_ROUTING
from router.providers.base import ProviderRequest, ProviderResponse


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_provider(response_text="ok", raises=None):
    """Create a mock provider that succeeds or raises."""
    provider = AsyncMock()
    if raises:
        provider.call.side_effect = raises
    else:
        provider.call.return_value = ProviderResponse(
            content=response_text,
            finish_reason="stop",
            input_tokens=10,
            output_tokens=20,
            provider_latency_ms=100,
        )
    return provider


def make_router_with_models(model_specs: list[dict]) -> LLMRouter:
    """
    Build a router with custom mock models.
    model_specs: [{"name": "gpt", "priority": 1, "raises": Exception("fail")}]
    """
    from router.llm_router import ModelConfig
    router = LLMRouter.__new__(LLMRouter)
    router.stats = __import__("router.llm_router", fromlist=["RouterStats"]).RouterStats()

    models = []
    for spec in model_specs:
        provider = make_provider(
            response_text=spec.get("response", f"response from {spec['name']}"),
            raises=spec.get("raises"),
        )
        m = ModelConfig(
            name=spec["name"],
            provider_name=spec.get("provider", spec["name"]),
            priority=spec["priority"],
            provider=provider,
        )
        models.append(m)

    router.models = sorted(models, key=lambda m: m.priority)
    return router


# ─── Tests: fallback behavior ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_first_model_success():
    """Router uses first available model when it succeeds."""
    router = make_router_with_models([
        {"name": "gpt",      "priority": 1, "response": "gpt answer"},
        {"name": "deepseek", "priority": 2, "response": "deepseek answer"},
    ])
    req = LLMRequest(system="s", messages=[{"role": "user", "content": "hi"}])
    result = await router.complete(req)

    assert result.model_used == "gpt"
    assert result.content == "gpt answer"
    assert result.fallback_used is False
    assert result.fallback_count == 0


@pytest.mark.asyncio
async def test_fallback_on_first_failure():
    """Router falls back to second model when first fails."""
    router = make_router_with_models([
        {"name": "gpt",      "priority": 1, "raises": Exception("500 server_error")},
        {"name": "deepseek", "priority": 2, "response": "deepseek answer"},
    ])
    req = LLMRequest(system="s", messages=[{"role": "user", "content": "hi"}])
    result = await router.complete(req)

    assert result.model_used == "deepseek"
    assert result.fallback_used is True
    assert result.fallback_count == 1
    assert "gpt" in str(result.errors_encountered)


@pytest.mark.asyncio
async def test_all_models_fail_raises():
    """Router raises RuntimeError when all models fail."""
    router = make_router_with_models([
        {"name": "gpt",      "priority": 1, "raises": Exception("quota")},
        {"name": "deepseek", "priority": 2, "raises": Exception("timeout")},
    ])
    req = LLMRequest(system="s", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(RuntimeError, match="All LLM models failed"):
        await router.complete(req)


@pytest.mark.asyncio
async def test_no_models_available_raises():
    """Router raises if all models are in FAILED state."""
    router = make_router_with_models([
        {"name": "gpt", "priority": 1},
    ])
    router.models[0].state = ModelState.FAILED

    req = LLMRequest(system="s", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(RuntimeError, match="No LLM models available"):
        await router.complete(req)


# ─── Tests: model health state ────────────────────────────────────────────────

def test_failed_model_skipped():
    """FAILED model is excluded from candidate list."""
    router = make_router_with_models([
        {"name": "gpt",      "priority": 1},
        {"name": "deepseek", "priority": 2},
    ])
    router.models[0].state = ModelState.FAILED

    available = router.get_available_models()
    names = [m.name for m in available]
    assert "gpt" not in names
    assert "deepseek" in names


def test_degraded_model_excluded_until_ttl():
    """DEGRADED model is unavailable until its TTL expires."""
    from datetime import datetime, timedelta
    router = make_router_with_models([{"name": "gpt", "priority": 1}])
    model = router.models[0]
    model.set_degraded(ttl_seconds=300)

    assert model.state == ModelState.DEGRADED
    assert model.is_available() is False


def test_degraded_model_recovers_after_ttl():
    """DEGRADED model becomes available after TTL."""
    from datetime import datetime, timedelta
    router = make_router_with_models([{"name": "gpt", "priority": 1}])
    model = router.models[0]
    model.degraded_until = datetime.utcnow() - timedelta(seconds=1)  # already expired
    model.state = ModelState.DEGRADED

    assert model.is_available() is True


# ─── Tests: task routing ──────────────────────────────────────────────────────

def test_task_routing_reorders_candidates():
    """Task type changes the model preference order."""
    router = make_router_with_models([
        {"name": "gpt-4o-mini",   "priority": 1},
        {"name": "deepseek-chat", "priority": 2},
        {"name": "qwen-turbo",    "priority": 3},
    ])
    # Default order: gpt → deepseek → qwen
    default = [m.name for m in router.get_available_models()]
    assert default[0] == "gpt-4o-mini"

    # Coding task: deepseek should be first
    coding = [m.name for m in router.get_available_models(task_type="coding")]
    assert coding[0] == "deepseek-chat"

    # Classification task: qwen should be first
    classification = [m.name for m in router.get_available_models(task_type="classification")]
    assert classification[0] == "qwen-turbo"


def test_unknown_task_type_uses_default_order():
    """Unknown task type falls back to default priority order."""
    router = make_router_with_models([
        {"name": "gpt-4o-mini", "priority": 1},
        {"name": "deepseek",    "priority": 2},
    ])
    result = [m.name for m in router.get_available_models(task_type="unknown_task")]
    assert result[0] == "gpt-4o-mini"


# ─── Tests: stats tracking ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_increment_on_success():
    router = make_router_with_models([{"name": "gpt", "priority": 1}])
    req = LLMRequest(system="s", messages=[{"role": "user", "content": "hi"}])
    await router.complete(req)

    assert router.stats.total_requests == 1
    assert router.stats.total_failures == 0
    assert router.stats.model_usage.get("gpt", 0) == 1


@pytest.mark.asyncio
async def test_stats_increment_on_all_fail():
    router = make_router_with_models([
        {"name": "gpt", "priority": 1, "raises": Exception("fail")},
    ])
    req = LLMRequest(system="s", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(RuntimeError):
        await router.complete(req)

    assert router.stats.total_failures == 1


# ─── Tests: response normalization ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_response_shape_is_consistent():
    """LLMResponse always has the same fields regardless of provider."""
    router = make_router_with_models([{"name": "gpt", "priority": 1}])
    req = LLMRequest(system="s", messages=[{"role": "user", "content": "hi"}])
    result = await router.complete(req)

    assert isinstance(result.content, str)
    assert isinstance(result.model_used, str)
    assert isinstance(result.fallback_used, bool)
    assert isinstance(result.latency_ms, int)
    assert isinstance(result.errors_encountered, list)
