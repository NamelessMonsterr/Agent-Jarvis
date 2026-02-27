"""
Planner Agent — interprets user intent, selects the right agent.
Never performs tasks itself. Pure routing.
"""
from agents.base import BaseAgent
from core.exceptions import AgentError
from core.logger import get_logger
from router.llm_router import LLMRouter
from schemas.api import PlannerDecision

log = get_logger("agent.planner")

SYSTEM_PROMPT = """You are the Planner for Jarvis, an AI developer assistant.
Your ONLY job: analyze the user message and decide which agent handles it.
You NEVER perform tasks yourself.

Available agents:
- research   → find papers, summarize research, explore topics
- dataset    → find datasets from HuggingFace/Kaggle
- github     → create devlog, commit progress, record today's work
- linkedin   → generate LinkedIn post from devlogs/research
- chat       → general conversation, questions, no specialist needed

Rules:
1. Respond ONLY with valid JSON
2. Choose exactly ONE agent
3. Extract a clean instruction for the agent

Format:
{"agent": "<name>", "intent": "<clean instruction>", "reasoning": "<one sentence>"}
"""

EXAMPLES = """
User: "find papers on RAG hallucination"
→ {"agent": "research", "intent": "find papers on RAG hallucination", "reasoning": "User wants academic research"}

User: "find datasets for that"
→ {"agent": "dataset", "intent": "find datasets for RAG hallucination research", "reasoning": "Continuation of prior research topic"}

User: "record today: built LLM router and tested fallback"
→ {"agent": "github", "intent": "create devlog for today: built LLM router and tested fallback", "reasoning": "User wants to document progress"}

User: "post to linkedin"
→ {"agent": "linkedin", "intent": "generate LinkedIn post from recent devlogs", "reasoning": "User wants to share progress publicly"}

User: "what is attention mechanism"
→ {"agent": "chat", "intent": "explain attention mechanism", "reasoning": "General technical question"}
"""


class PlannerAgent(BaseAgent):
    name = "planner"
    description = "Routes user intent to the correct agent"

    async def run(self, user_input: str, session_id: str, user_id: str, **kwargs) -> tuple[str, str]:
        start_time = self.logger.start(session_id, user_input)

        context = self._get_context_str(session_id)
        messages = [{
            "role": "user",
            "content": (
                f"Examples:\n{EXAMPLES}\n\n"
                f"Conversation so far:\n{context}\n\n"
                f"New message: {user_input}"
            ),
        }]

        response = await self.llm(
            system=SYSTEM_PROMPT,
            task_type="classification",
            messages=messages,
            max_tokens=150,
            temperature=0.1,
            json_mode=True,
        )

        try:
            raw = LLMRouter.extract_json(response.content)
            decision = PlannerDecision.from_llm(raw)
            self.logger.success(start_time, response.model_used, f"→ [{decision.agent}]")
            return decision.agent, decision.intent or user_input
        except (ValueError, Exception) as e:
            log.warning(f"Planner parse failed — defaulting to chat. Raw: {response.content[:100]!r} | err: {e}")
            return "chat", user_input


planner_agent = PlannerAgent()
