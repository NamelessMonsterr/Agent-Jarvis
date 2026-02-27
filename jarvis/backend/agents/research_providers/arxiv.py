"""
ArXiv provider — searches arxiv.org API, normalizes output via PaperSchema adapter.
"""
import httpx
import xml.etree.ElementTree as ET
from core.exceptions import ProviderError
from core.logger import get_logger
from agents.research_providers.normalizer import normalize_papers

log = get_logger("research.arxiv")

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivProvider:

    async def search(self, query: str, max_results: int = 8) -> list:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(ARXIV_API, params=params)
            resp.raise_for_status()
            raws = self._parse_to_dicts(resp.text)
            papers = normalize_papers(raws, source="arxiv")
            log.info(f"ArXiv returned {len(papers)} papers for '{query}'")
            return papers
        except httpx.TimeoutException:
            raise ProviderError("arxiv", "Request timed out", status_code=408)
        except httpx.HTTPStatusError as e:
            raise ProviderError("arxiv", f"HTTP {e.response.status_code}", status_code=e.response.status_code)
        except Exception as e:
            log.warning(f"ArXiv search failed: {e}")
            return []  # Non-fatal: other provider may still succeed

    def _parse_to_dicts(self, xml_text: str) -> list[dict]:
        root = ET.fromstring(xml_text)
        results = []
        for entry in root.findall("atom:entry", NS):
            def txt(tag): el = entry.find(f"atom:{tag}", NS); return el.text.strip() if el is not None and el.text else ""

            id_url = txt("id")
            arxiv_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else id_url

            authors = []
            for a in entry.findall("atom:author", NS):
                n = a.find("atom:name", NS)
                if n is not None and n.text:
                    authors.append(n.text.strip())

            doi = None
            for link in entry.findall("atom:link", NS):
                if link.get("title") == "doi":
                    doi = link.get("href", "").replace("https://doi.org/", "")

            published = txt("published")
            year = int(published[:4]) if published and len(published) >= 4 else None

            results.append({
                "title":          txt("title"),
                "abstract":       txt("summary"),
                "authors":        authors,
                "year":           year,
                "url":            f"https://arxiv.org/abs/{arxiv_id}",
                "doi":            doi or arxiv_id,
                "citation_count": 0,
            })
        return results
