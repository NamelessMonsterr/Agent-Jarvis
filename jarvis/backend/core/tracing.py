"""
core/tracing.py — Agent execution tracing for Jarvis.

Captures the execution timeline:
  Planner (50ms)
   ├─ ResearchAgent (2.1s) → gpt-4o-mini
   └─ Router fallback: deepseek-chat

Usage:
  trace = TraceContext(session_id, request_id)
  with trace.span("research") as span:
      result = await agent.run(...)
      span.model = "gpt-4o-mini"
  trace.to_dict()  # → serializable timeline
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TraceSpan:
    """One step in the execution timeline."""
    agent: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: int = 0
    status: str = "running"    # running | success | failed | timeout
    model: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def finish(self, status: str = "success", model: str = None, error: str = None) -> None:
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)
        self.status = status
        if model:
            self.model = model
        if error:
            self.error = error

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "model": self.model,
            "error": self.error,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


class TraceContext:
    """
    Collects trace spans across an entire request.
    Passed through planner → agent → router to build execution timeline.
    """

    def __init__(self, session_id: str, request_id: str = ""):
        self.session_id = session_id
        self.request_id = request_id
        self.spans: list[TraceSpan] = []
        self._start_time = time.time()

    def start_span(self, agent: str, **metadata) -> TraceSpan:
        """Start a new span. Remember to call .finish() when done."""
        span = TraceSpan(agent=agent, metadata=metadata)
        self.spans.append(span)
        return span

    @property
    def total_duration_ms(self) -> int:
        return int((time.time() - self._start_time) * 1000)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "total_duration_ms": self.total_duration_ms,
            "spans": [s.to_dict() for s in self.spans],
        }

    def to_timeline_str(self) -> str:
        """Human-readable execution timeline for logs."""
        lines = [f"Trace [{self.request_id}] total={self.total_duration_ms}ms"]
        for s in self.spans:
            icon = {"success": "✓", "failed": "✗", "timeout": "⏱"}.get(s.status, "•")
            model_str = f" → {s.model}" if s.model else ""
            error_str = f" | err={s.error[:40]}" if s.error else ""
            lines.append(f"  {icon} {s.agent} ({s.duration_ms}ms){model_str}{error_str}")
        return "\n".join(lines)
