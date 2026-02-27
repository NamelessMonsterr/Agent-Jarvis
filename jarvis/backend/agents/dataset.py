"""
Dataset Agent v3 — HF + Kaggle → LLM rank → schemas format.
Configurable scoring weights. No inline string building.
"""
import asyncio
import hashlib
from agents.base import BaseAgent
from agents.dataset_providers.huggingface import HuggingFaceProvider
from agents.dataset_providers.kaggle import KaggleProvider
from core.config import settings
from core.exceptions import AgentError
from memory.models import KnowledgeEntry, ChunkMetadata
from memory.service import sanitize_for_llm
from router.llm_router import LLMRouter
from schemas.api import DatasetRanking
from schemas.response import format_datasets, format_datasets_empty
from core.logger import get_logger

log = get_logger(__name__)

# Explicit scoring weights — tune these without touching agent logic
DATASET_SCORE_WEIGHTS = {
    "relevance":   0.40,
    "popularity":  0.40,
    "usability":   0.20,
}

RANK_PROMPT = f"""You are a dataset evaluator for ML research.

Score each dataset across three criteria and return the top 3:

Scoring weights (must add to 100):
- relevance  = {int(DATASET_SCORE_WEIGHTS['relevance']*100)} pts  (how well it matches the research need)
- popularity = {int(DATASET_SCORE_WEIGHTS['popularity']*100)} pts (downloads/likes = community validation)
- usability  = {int(DATASET_SCORE_WEIGHTS['usability']*100)} pts  (size, format, documentation)

Respond in JSON only:
{{
  "ranked": [
    {{
      "name": "...",
      "source": "huggingface|kaggle",
      "score": <0-100>,
      "relevance_reason": "...",
      "limitations": "...",
      "use_for": "..."
    }}
  ],
  "recommendation": "..."
}}
"""


class DatasetAgent(BaseAgent):
    name = "dataset"
    description = "Searches HuggingFace + Kaggle, ranks with LLM, caches results in memory"

    def __init__(self):
        super().__init__()
        self.hf = HuggingFaceProvider()
        self.kaggle = KaggleProvider()

    async def run(self, user_input: str, session_id: str, user_id: str, **kwargs) -> str:
        start_time = self.logger.start(session_id, user_input)

        # Cache check
        cache_key = hashlib.sha256(f"{user_id}:{user_input.lower().strip()}".encode()).hexdigest()[:12]
        cached = self.memory.tag_search([f"cache:{cache_key}"], user_id)
        if cached:
            self.logger.skip(f"cache hit for '{user_input[:40]}'")
            self._save_agent_message(session_id, cached[0].content)
            return cached[0].content

        # Cross-agent context from research memory
        prior = await self.memory.search_knowledge(user_input, user_id, k=3, filter_type="research")
        enriched_query = user_input
        if prior:
            topics = [e.metadata.topic for e in prior if e.metadata.topic]
            if topics:
                enriched_query = f"{user_input} {' '.join(topics[:2])}"
            self.logger.info(f"Enriched query with {len(prior)} research entries")

        # Parallel provider search
        try:
            hf_results, kg_results = await asyncio.gather(
                self.hf.search(enriched_query, max_results=10),
                self.kaggle.search(enriched_query, max_results=10),
            )
        except Exception as e:
            raise AgentError(self.name, f"Dataset provider search failed: {e}")

        all_datasets = hf_results + kg_results
        self.logger.info(f"{len(all_datasets)} datasets found ({len(hf_results)} HF, {len(kg_results)} Kaggle)")

        if not all_datasets:
            out = format_datasets_empty(user_input)
            self._save_agent_message(session_id, out)
            return out

        # LLM ranks the candidates
        summary = "\n".join(
            f"{i}. [{ds.source}] {ds.name} | dl={ds.downloads:,} likes={ds.likes} size={ds.size}"
            for i, ds in enumerate(all_datasets[:15], 1)
        )
        response = await self.llm(
            system=RANK_PROMPT,
            task_type="reasoning",
            messages=[{"role": "user", "content": f"Topic: {sanitize_for_llm(user_input)}\n\nDatasets:\n{summary}"}],
            max_tokens=700,
            temperature=0.2,
            json_mode=True,
        )

        try:
            raw = LLMRouter.extract_json(response.content)
            ranking = DatasetRanking.from_llm(raw)
        except (ValueError, Exception) as e:
            raise AgentError(self.name, f"LLM returned unparseable ranking response: {e}")

        # Enrich validated ranked items with full provider metadata
        ds_map = {ds.name: ds for ds in all_datasets}
        ranked = []
        for item in ranking.ranked[:3]:
            ds = ds_map.get(item.name)
            if ds:
                item.url = ds.url
                item.downloads = ds.downloads
                item.size = ds.size
            ranked.append(item.model_dump())

        out = format_datasets(ranked, ranking.recommendation, user_input, response.model_used)

        # Cache result
        await self.memory.store_knowledge(KnowledgeEntry(
            user_id=user_id, agent="dataset",
            title=f"Dataset search: {user_input[:60]}",
            content=out,
            summary=f"Top {len(ranked)} datasets for: {user_input}",
            tags=[f"cache:{cache_key}", "dataset-search"],
            metadata=ChunkMetadata(type="dataset", agent="dataset", topic=user_input[:80]),
            session_id=session_id,
        ))

        self.logger.success(start_time, response.model_used, f"top {len(ranked)} returned")
        self._save_agent_message(session_id, out)
        return out


dataset_agent = DatasetAgent()
