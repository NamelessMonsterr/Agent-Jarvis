"""
Kaggle dataset provider.
Uses the Kaggle public API. Requires KAGGLE_USERNAME + KAGGLE_KEY in .env.
If credentials not set, returns empty list gracefully (HuggingFace still runs).
"""
import httpx
import base64
from memory.models import Dataset
from core.config import settings
from core.logger import get_logger

log = get_logger("dataset.kaggle")

KAGGLE_API = "https://www.kaggle.com/api/v1/datasets/list"


class KaggleProvider:

    def __init__(self):
        username = getattr(settings, "KAGGLE_USERNAME", None)
        key = getattr(settings, "KAGGLE_KEY", None)
        if username and key:
            token = base64.b64encode(f"{username}:{key}".encode()).decode()
            self._auth_header = f"Basic {token}"
        else:
            self._auth_header = None

    async def search(self, query: str, max_results: int = 10) -> list[Dataset]:
        if not self._auth_header:
            log.debug("Kaggle credentials not configured — skipping")
            return []

        params = {"search": query, "pageSize": max_results, "sortBy": "hotness"}
        headers = {"Authorization": self._auth_header}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(KAGGLE_API, params=params, headers=headers)
            if resp.status_code == 401:
                log.warning("Kaggle auth failed — check KAGGLE_USERNAME and KAGGLE_KEY")
                return []
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception as e:
            log.warning(f"Kaggle search failed for '{query}': {e}")
            return []

    def _parse(self, items: list[dict]) -> list[Dataset]:
        datasets = []
        for item in items:
            ref = item.get("ref", "")
            title = item.get("title", ref)
            size_bytes = item.get("totalBytes", 0)
            size_label = self._fmt_size(size_bytes)

            datasets.append(Dataset(
                name=title,
                source="kaggle",
                url=f"https://www.kaggle.com/datasets/{ref}",
                downloads=item.get("downloadCount") or 0,
                likes=item.get("voteCount") or 0,
                size=size_label,
                format=item.get("licenseName", "unknown"),
            ))
        return datasets

    def _fmt_size(self, b: int) -> str:
        if b == 0:
            return "unknown"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} PB"
