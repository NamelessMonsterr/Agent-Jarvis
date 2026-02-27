from agents.research import research_agent
from agents.dataset import dataset_agent
from agents.github import github_agent
from agents.linkedin import linkedin_agent
from core.config import settings
from core.logger import get_logger
from agents.base import BaseAgent

log = get_logger(__name__)


def _build_registry() -> dict[str, BaseAgent]:
    """Build agent registry respecting feature flags."""
    registry = {}
    if settings.ENABLE_RESEARCH:
        registry["research"] = research_agent
    if settings.ENABLE_DATASET:
        registry["dataset"] = dataset_agent
    if settings.ENABLE_GITHUB:
        registry["github"] = github_agent
    if settings.ENABLE_LINKEDIN:
        registry["linkedin"] = linkedin_agent
    log.info(f"Agent registry: {list(registry.keys())}")
    return registry


AGENT_REGISTRY: dict[str, BaseAgent] = _build_registry()


def get_agent(name: str) -> BaseAgent | None:
    return AGENT_REGISTRY.get(name)
