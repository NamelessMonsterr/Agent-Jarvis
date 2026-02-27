"""
core/logger.py — Single unified logger for all Jarvis components.
Import get_logger() or AgentLogger everywhere. Never use print().
"""
import logging
import time
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Silence noisy third-party loggers
for lib in ("httpx", "httpcore", "openai", "supabase", "urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"jarvis.{name}")


class AgentLogger:
    """Structured logger attached to each agent. Measures latency automatically."""

    def __init__(self, agent_name: str):
        self._log = get_logger(f"agent.{agent_name}")
        self.agent_name = agent_name

    def start(self, session_id: str, intent: str) -> float:
        self._log.info(f"START  | session={session_id[:8]} | intent={intent[:80]!r}")
        return time.time()

    def success(self, start_time: float, model: str, detail: str = "") -> None:
        elapsed = int((time.time() - start_time) * 1000)
        self._log.info(f"OK     | latency={elapsed}ms | model={model} | {detail}")

    def failure(self, error: str, detail: str = "") -> None:
        self._log.error(f"FAIL   | error={error!r} | {detail}")

    def skip(self, reason: str) -> None:
        self._log.info(f"SKIP   | reason={reason}")

    def info(self, msg: str) -> None:
        self._log.info(msg)

    def warn(self, msg: str) -> None:
        self._log.warning(msg)

    def debug(self, msg: str) -> None:
        self._log.debug(msg)


# Named module-level loggers for non-agent components
router_log  = get_logger("router")
memory_log  = get_logger("memory")
api_log     = get_logger("api")
db_log      = get_logger("db")
