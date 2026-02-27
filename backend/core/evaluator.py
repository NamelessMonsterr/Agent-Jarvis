"""
core/evaluator.py — Response evaluation layer for Jarvis.

Measures response quality after each agent execution.
Stores metrics alongside agent_runs for production monitoring.

Usage:
    evaluator.evaluate(
        query="find papers on RAG",
        response="## Research Results...",
        agent_name="research",
        model_used="gpt-4o-mini",
        duration_ms=2100,
        session_id="abc123",
        user_id="user-1",
    )
"""
from core.logger import get_logger

log = get_logger("evaluator")


class ResponseEvaluator:
    """
    Lightweight evaluator that scores responses based on heuristics.
    No LLM call — fast enough to run on every response.
    """

    def evaluate(
        self,
        query: str,
        response: str,
        agent_name: str,
        model_used: str = "",
        duration_ms: int = 0,
        session_id: str = "",
        user_id: str = "",
    ) -> dict:
        """
        Evaluate a response and return quality metrics.
        Returns dict with scores — higher is better.
        """
        scores = {
            "has_content": bool(response and len(response.strip()) > 20),
            "response_length": len(response),
            "is_error": response.strip().startswith("⚠️") or "failed" in response.lower()[:50],
            "is_fallback": "fallback" in response.lower()[:100],
            "completeness": self._score_completeness(response, agent_name),
            "latency_grade": self._grade_latency(duration_ms),
        }

        # Composite usefulness score (0-100)
        usefulness = 0
        if scores["has_content"]:
            usefulness += 40
        if not scores["is_error"]:
            usefulness += 30
        usefulness += scores["completeness"] * 20
        usefulness += scores["latency_grade"] * 10
        scores["usefulness_score"] = min(100, usefulness)

        # Store in DB (fire-and-forget)
        self._store_evaluation(scores, query, agent_name, model_used, session_id, user_id)

        return scores

    def _score_completeness(self, response: str, agent_name: str) -> float:
        """Score how complete the response is for its agent type (0.0-1.0)."""
        if agent_name == "research":
            # Research should have papers, synthesis, insights
            markers = ["##", "Papers", "Synthesis", "Insight", "http"]
            found = sum(1 for m in markers if m.lower() in response.lower())
            return min(1.0, found / 3)
        elif agent_name == "dataset":
            markers = ["##", "Score:", "Downloads", "http"]
            found = sum(1 for m in markers if m.lower() in response.lower())
            return min(1.0, found / 3)
        elif agent_name == "github":
            markers = ["Devlog", "Tasks", "✅", "Commit"]
            found = sum(1 for m in markers if m in response)
            return min(1.0, found / 2)
        elif agent_name == "linkedin":
            markers = ["##", "#", "hook", "DRAFT"]
            found = sum(1 for m in markers if m.lower() in response.lower())
            return min(1.0, found / 2)
        return 0.5  # default for chat

    def _grade_latency(self, duration_ms: int) -> float:
        """Grade latency (0.0-1.0). Under 2s = great, over 10s = poor."""
        if duration_ms <= 0:
            return 0.5
        if duration_ms < 2000:
            return 1.0
        if duration_ms < 5000:
            return 0.7
        if duration_ms < 10000:
            return 0.4
        return 0.1

    def _store_evaluation(
        self, scores: dict, query: str, agent_name: str,
        model_used: str, session_id: str, user_id: str,
    ) -> None:
        """Store evaluation metrics (best-effort, never blocks)."""
        try:
            from db.client import get_supabase
            db = get_supabase()
            db.table("agent_runs").update({
                "evaluation": scores,
            }).eq("session_id", session_id).eq(
                "agent_name", agent_name
            ).eq("status", "success").order(
                "started_at", desc=True
            ).limit(1).execute()
        except Exception:
            pass  # Never let evaluation errors affect the response

        log.debug(
            f"Eval: agent={agent_name} usefulness={scores.get('usefulness_score', '?')} "
            f"latency={scores.get('latency_grade', '?')}"
        )


# Singleton
evaluator = ResponseEvaluator()
