from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Jarvis"
    DEBUG: bool = False
    DEFAULT_USER_ID: str = "local-dev-user"

    # Security
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    RATE_LIMIT_PER_MINUTE: int = 30
    # Auth — set AUTH_REQUIRED=true in production after testing locally
    AUTH_REQUIRED: bool = False
    JWT_SECRET: str = ""              # generate with: python -c "import secrets; print(secrets.token_hex(32))"
    AUTH_TOKEN_SECRET: str = ""       # optional: passphrase required to issue tokens

    # LLM Providers
    OPENAI_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    QWEN_API_KEY: Optional[str] = None
    GROK_API_KEY: Optional[str] = None

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""

    # GitHub
    GITHUB_TOKEN: Optional[str] = None
    GITHUB_REPO: Optional[str] = None
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None
    GITHUB_DEVLOG_BRANCH: str = "jarvis/devlog"
    GITHUB_DRY_RUN: bool = False
    GITHUB_MIN_DIFF_CHARS: int = 50

    # LinkedIn
    LINKEDIN_CLIENT_ID: Optional[str] = None
    LINKEDIN_CLIENT_SECRET: Optional[str] = None

    # Memory / RAG
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    MAX_CONTEXT_MESSAGES: int = 10
    SESSION_TTL_HOURS: int = 2
    RAG_CHUNK_SIZE: int = 600
    RAG_CHUNK_OVERLAP: int = 100
    RAG_TOP_K: int = 5
    RECENCY_WEIGHT: float = 0.2

    # Agent Execution
    AGENT_TIMEOUT_S: int = 60
    AGENT_MAX_RETRIES: int = 1

    # Feature Flags — disable agents without code changes
    ENABLE_RESEARCH: bool = True
    ENABLE_DATASET: bool = True
    ENABLE_GITHUB: bool = True
    ENABLE_LINKEDIN: bool = False    # disabled by default until OAuth is ready

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


# ─── Startup validation ──────────────────────────────────────────────────────

def validate_config() -> list[str]:
    """Validate critical config at startup. Returns list of warnings."""
    warnings = []
    if settings.AUTH_REQUIRED and not settings.JWT_SECRET:
        raise ValueError(
            "FATAL: AUTH_REQUIRED=true but JWT_SECRET is empty. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    if not settings.SUPABASE_URL:
        warnings.append("SUPABASE_URL not set — memory features will fail")
    if not any([settings.OPENAI_API_KEY, settings.DEEPSEEK_API_KEY,
                settings.QWEN_API_KEY, settings.GROK_API_KEY]):
        warnings.append("No LLM API keys configured — only Ollama will be available")
    return warnings
