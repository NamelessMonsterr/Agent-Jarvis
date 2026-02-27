"""
HuggingFace dataset provider — searches the HF Hub API.
No API key required for public datasets.
"""
import httpx
from memory.models import Dataset
from core.logger import get_logger

log = get_logger("dataset.huggingface")

HF_API = "https://huggingface.co/api/datasets"


class HuggingFaceProvider:

    async def search(self, query: str, max_results: int = 10) -> list[Dataset]:
        params = {
            "search": query,
            "limit": max_results,
            "full": "true",
            "sort": "downloads",
            "direction": -1,
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(HF_API, params=params)
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception as e:
            log.warning(f"HuggingFace search failed for '{query}': {e}")
            return []

    def _parse(self, items: list[dict]) -> list[Dataset]:
        datasets = []
        for item in items:
            dataset_id = item.get("id", "")
            if not dataset_id:
                continue

            # Infer size from cardData if present
            card = item.get("cardData") or {}
            size_label = "unknown"
            dataset_info = card.get("dataset_info")
            if isinstance(dataset_info, dict):
                splits = dataset_info.get("splits", [])
                total = sum(s.get("num_examples", 0) for s in splits if isinstance(s, dict))
                if total:
                    size_label = f"{total:,} examples"

            datasets.append(Dataset(
                name=dataset_id,
                source="huggingface",
                url=f"https://huggingface.co/datasets/{dataset_id}",
                huggingface_id=dataset_id,
                downloads=item.get("downloads") or 0,
                likes=item.get("likes") or 0,
                size=size_label,
                format=", ".join((item.get("tags") or [])[:3]) or "unknown",
            ))
        return datasets
