"""
Semantic Scholar provider — searches S2 API, normalizes via PaperSchema adapter.
"""
import httpx
from core.exceptions import ProviderError
from core.logger import get_logger
from agents.research_providers.normalizer import normalize_papers

log = get_logger("research.semantic_scholar")

S2_API    = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,authors,year,externalIds,url,citationCount"


class SemanticScholarProvider:

    async def search(self, query: str, max_results: int = 8) -> list:
        params = {"query": query, "limit": max_results, "fields": S2_FIELDS}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(S2_API, params=params)

            if resp.status_code == 429:
                log.warning("Semantic Scholar rate limited — returning empty")
                return []
            resp.raise_for_status()

            raws = self._parse_to_dicts(resp.json().get("data", []))
            papers = normalize_papers(raws, source="semantic_scholar")
            log.info(f"Semantic Scholar returned {len(papers)} papers for '{query}'")
            return papers

        except httpx.TimeoutException:
            raise ProviderError("semantic_scholar", "Request timed out", status_code=408)
        except httpx.HTTPStatusError as e:
            raise ProviderError("semantic_scholar", f"HTTP {e.response.status_code}", status_code=e.response.status_code)
        except Exception as e:
            log.warning(f"Semantic Scholar search failed: {e}")
            return []

    def _parse_to_dicts(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            ext = item.get("externalIds") or {}
            results.append({
                "title":          item.get("title", ""),
                "abstract":       item.get("abstract") or "",
                "authors":        [a.get("name", "") for a in (item.get("authors") or [])],
                "year":           item.get("year"),
                "url":            item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId','')}",
                "doi":            ext.get("DOI") or ext.get("ArXiv"),
                "citation_count": item.get("citationCount") or 0,
            })
        return results
