from fastapi import APIRouter
from core.config import settings
from core.logger import get_logger
from router.llm_router import llm_router

router = APIRouter(tags=["system"])
log = get_logger("api.health")


@router.get("/health")
async def health():
    log.debug("Health check called")
    return {"status": "ok", "app": settings.APP_NAME, "version": "3.4.0"}


@router.get("/status")
async def status():
    """Returns system status including live LLM model availability."""
    models = []
    for m in llm_router.models:
        models.append({
            "name": m.name,
            "provider": m.provider_name,
            "priority": m.priority,
            "state": m.state.value,
            "available": m.is_available(),
            "has_key": m.provider_name != "ollama",
            "total_calls": m.total_calls,
            "total_failures": m.total_failures,
            "failure_rate": (
                round(m.total_failures / m.total_calls, 2)
                if m.total_calls > 0 else 0.0
            ),
        })

    available_count = sum(1 for m in models if m["available"])
    log.info(f"Status check: {available_count}/{len(models)} models available")

    return {
        "status": "ok" if available_count > 0 else "degraded",
        "version": "3.4.0",
        "app": settings.APP_NAME,
        "llm_models": models,
        "available_models": available_count,
        "router_stats": llm_router.stats.to_dict(),
        "github_configured": bool(settings.GITHUB_TOKEN and settings.GITHUB_REPO),
        "github_dry_run": settings.GITHUB_DRY_RUN,
        "supabase_configured": bool(settings.SUPABASE_URL),
    }
