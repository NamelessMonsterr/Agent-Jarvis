"""
Research Agent v3 — real API search → normalizer → LLM synthesize → schemas format.
"""
import asyncio
from agents.base import BaseAgent
from agents.research_providers.arxiv import ArxivProvider
from agents.research_providers.semantic_scholar import SemanticScholarProvider
from core.exceptions import AgentError
from memory.models import KnowledgeEntry, ChunkMetadata
from memory.service import sanitize_for_llm
from router.llm_router import LLMRouter
from schemas.api import ResearchSynthesis
from schemas.response import format_research, format_research_no_results
from core.logger import get_logger

log = get_logger(__name__)

SYNTHESIZE_PROMPT = """You are a research analyst given real papers retrieved from ArXiv and Semantic Scholar.

Synthesize them. Respond in JSON only:
{
  "synthesis": "<2-3 paragraph overview of the research landscape>",
  "key_insights": ["insight 1", "insight 2"],
  "open_problems": ["problem 1"],
  "datasets_mentioned": ["dataset name 1"],
  "tags": ["tag1", "tag2"]
}
"""


class ResearchAgent(BaseAgent):
    name = "research"
    description = "Fetches real papers from ArXiv + Semantic Scholar, synthesizes with LLM"

    def __init__(self):
        super().__init__()
        self.arxiv = ArxivProvider()
        self.s2 = SemanticScholarProvider()

    async def run(self, user_input: str, session_id: str, user_id: str, **kwargs) -> str:
        start_time = self.logger.start(session_id, user_input)

        # Parallel API search
        try:
            arxiv_papers, s2_papers = await asyncio.gather(
                self.arxiv.search(user_input, max_results=6),
                self.s2.search(user_input, max_results=6),
            )
        except Exception as e:
            raise AgentError(self.name, f"Provider search failed: {e}")

        # Deduplicate by content hash
        all_papers = self._dedup(arxiv_papers + s2_papers)
        self.logger.info(f"{len(all_papers)} unique papers ({len(arxiv_papers)} ArXiv, {len(s2_papers)} S2)")

        # Fallback to LLM-only if no API results
        if not all_papers:
            response = await self.llm(
                system="Summarize the state of research on this topic accurately based on your training knowledge.",
                task_type="reasoning",
                messages=[{"role": "user", "content": f"Topic: {sanitize_for_llm(user_input)}"}],
                max_tokens=600,
            )
            out = format_research_no_results(user_input, response.content, response.model_used)
            self._save_agent_message(session_id, out)
            return out

        # Prior knowledge context
        prior = await self.memory.search_knowledge(user_input, user_id, k=3, filter_type="research")
        prior_ctx = ""
        if prior:
            known = [e.title for e in prior]
            prior_ctx = f"\n\nAlready in memory (avoid repeating): {', '.join(known)}"

        papers_text = "\n\n".join(p.to_context_str() for p in all_papers[:10])
        response = await self.llm(
            system=SYNTHESIZE_PROMPT,
            task_type="reasoning",
            messages=[{"role": "user", "content": f"Topic: {sanitize_for_llm(user_input)}\n\nPapers:\n{papers_text}{prior_ctx}"}],
            max_tokens=1200,
            temperature=0.3,
            json_mode=True,
        )

        try:
            raw = LLMRouter.extract_json(response.content)
            synthesis = ResearchSynthesis.from_llm(raw)
        except (ValueError, Exception) as e:
            log.warning(f"Synthesis parse failed, using fallback: {e}")
            synthesis = ResearchSynthesis(synthesis=response.content)

        # Store each unique paper
        stored = 0
        for paper in all_papers[:8]:
            entry = KnowledgeEntry(
                user_id=user_id,
                agent="research",
                title=paper.title,
                content=f"{paper.abstract}\n\nAuthors: {', '.join(paper.authors)}\nURL: {paper.url}",
                summary=paper.abstract[:300],
                source_url=paper.url,
                tags=synthesis.tags + [paper.source],
                metadata=ChunkMetadata(type="research", agent="research", topic=user_input[:80]),
                content_hash=paper.content_hash(),
                session_id=session_id,
            )
            ids = await self.memory.store_knowledge(entry)
            if ids:
                stored += 1

        # Store synthesis
        await self.memory.store_knowledge(KnowledgeEntry(
            user_id=user_id, agent="research",
            title=f"Synthesis: {user_input[:60]}",
            content=synthesis.synthesis,
            summary=synthesis.synthesis[:300],
            tags=synthesis.tags,
            metadata=ChunkMetadata(type="research", agent="research", topic=user_input[:80]),
            session_id=session_id,
        ))

        self.logger.success(start_time, response.model_used, f"{stored} papers stored")

        # Formatting handled by schema — agent returns nothing inline
        out = format_research(all_papers, synthesis.model_dump(), response.model_used)
        self._save_agent_message(session_id, out)
        return out

    def _dedup(self, papers: list) -> list:
        seen, unique = set(), []
        for p in papers:
            h = p.content_hash()
            if h not in seen:
                seen.add(h)
                unique.append(p)
        return unique


research_agent = ResearchAgent()
