"""
core/exceptions.py — All Jarvis exceptions live here.
Silent `except Exception: return None` is banned. Use these instead.
"""


class JarvisError(Exception):
    """Base exception for all Jarvis errors."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}


class AgentError(JarvisError):
    """Raised when an agent fails to complete its task."""
    def __init__(self, agent_name: str, reason: str, context: dict = None):
        super().__init__(f"[{agent_name}] {reason}", context)
        self.agent_name = agent_name
        self.reason = reason


class ProviderError(JarvisError):
    """Raised when an external API provider fails."""
    def __init__(self, provider: str, reason: str, status_code: int = None):
        super().__init__(f"[{provider}] {reason}")
        self.provider = provider
        self.reason = reason
        self.status_code = status_code


class JarvisMemoryError(JarvisError):
    """Raised when a memory read/write operation fails."""
    pass


class RouterExhaustedError(JarvisError):
    """Raised when all LLM models fail."""
    def __init__(self, errors: list[str]):
        super().__init__(f"All LLM models failed: {errors}")
        self.errors = errors


class RateLimitError(JarvisError):
    """Raised when a session exceeds the request rate limit."""
    pass


class ConfigError(JarvisError):
    """Raised on missing or invalid configuration."""
    pass
