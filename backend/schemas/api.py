"""
schemas/api.py — Canonical request/response contracts for the API layer.
Every endpoint returns one of these. No ad-hoc dicts at the boundary.

Frontend depends on this shape. Do not change field names without a version bump.
"""
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal


# ─── Standard envelope ───────────────────────────────────────────────────────

class JarvisResponse(BaseModel):
    """
    Single envelope wrapping every successful API response.
    Frontend unpacks `data`; `meta` carries operational context.
    """
    ok: bool = True
    data: Any
    meta: "ResponseMeta"

    @classmethod
    def success(cls, data: Any, **meta_kwargs) -> "JarvisResponse":
        return cls(ok=True, data=data, meta=ResponseMeta(**meta_kwargs))

    @classmethod
    def error(cls, message: str, code: str = "error") -> "JarvisResponse":
        return cls(ok=False, data=None, meta=ResponseMeta(error=message, error_code=code))


class ResponseMeta(BaseModel):
    session_id: Optional[str] = None
    agent_used: Optional[str] = None
    model_used: Optional[str] = None
    fallback_used: bool = False
    latency_ms: Optional[int] = None
    request_id: Optional[str] = None
    trace: Optional[dict] = None     # execution timeline from TraceContext
    error: Optional[str] = None
    error_code: Optional[str] = None


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatData(BaseModel):
    response: str
    session_id: str
    agent_used: str
    model_used: str
    fallback_used: bool


# ─── Session ─────────────────────────────────────────────────────────────────

class SessionHistoryData(BaseModel):
    session_id: str
    message_count: int
    messages: list[dict]


# ─── Tone Profile ─────────────────────────────────────────────────────────────

class ToneRequest(BaseModel):
    tone: Literal["builder", "technical", "casual", "professional"] = "builder"
    length: Literal["short", "medium", "long"] = "medium"
    audience: Literal["tech", "general", "founders"] = "tech"
    avoid_phrases: list[str] = Field(default_factory=list)


# ─── Memory ──────────────────────────────────────────────────────────────────

class MemoryCompressData(BaseModel):
    compressed: int
    user_id: str


# ─── Error responses (used by exception handlers) ────────────────────────────

class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    error_code: str
    detail: Optional[str] = None


# ─── LLM JSON Response Schemas ───────────────────────────────────────────────
# Every agent parses LLM JSON through one of these instead of raw data.get().


class PlannerDecision(BaseModel):
    agent: str = "chat"
    intent: str = ""
    reasoning: str = ""

    @classmethod
    def from_llm(cls, raw: dict) -> "PlannerDecision":
        return cls.model_validate(raw)


class ResearchSynthesis(BaseModel):
    synthesis: str = ""
    key_insights: list[str] = Field(default_factory=list)
    open_problems: list[str] = Field(default_factory=list)
    datasets_mentioned: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_llm(cls, raw: dict) -> "ResearchSynthesis":
        return cls.model_validate(raw)


class DatasetRankItem(BaseModel):
    name: str = ""
    source: str = "huggingface"
    score: int = 0
    relevance_reason: str = ""
    limitations: str = ""
    use_for: str = ""
    # Enriched after LLM ranking — not from LLM
    url: str = ""
    downloads: int = 0
    size: str = "unknown"


class DatasetRanking(BaseModel):
    ranked: list[DatasetRankItem] = Field(default_factory=list)
    recommendation: str = ""

    @classmethod
    def from_llm(cls, raw: dict) -> "DatasetRanking":
        return cls.model_validate(raw)


class DevlogParse(BaseModel):
    summary: str = ""
    tasks: list[dict] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    mood: Optional[int] = None

    @classmethod
    def from_llm(cls, raw: dict) -> "DevlogParse":
        return cls.model_validate(raw)


class LinkedInPostDraft(BaseModel):
    post: str = ""
    hook: str = ""
    hashtags: list[str] = Field(default_factory=list)
    word_count: int = 0

    @classmethod
    def from_llm(cls, raw: dict) -> "LinkedInPostDraft":
        return cls.model_validate(raw)
