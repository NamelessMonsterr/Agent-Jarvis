"""
Research provider normalizer.
All provider results pass through normalize_paper() before reaching the agent.
Keeps the agent immune to provider-specific quirks.
"""
from memory.models import Paper
from core.logger import get_logger

log = get_logger("research.normalizer")

_MAX_ABSTRACT = 1200
_MAX_TITLE = 300
_MAX_AUTHORS = 10


def normalize_paper(raw: dict, source: str) -> Paper | None:
    """
    Convert a raw provider dict → clean Paper object.
    Returns None if the entry is too incomplete to be useful.
    """
    title = _clean_str(raw.get("title", ""))
    if not title or len(title) < 5:
        log.debug(f"Discarding paper with empty/short title from {source}")
        return None

    abstract = _clean_str(raw.get("abstract", ""))
    authors = _clean_authors(raw.get("authors", []))
    year = _clean_year(raw.get("year"))
    url = _clean_str(raw.get("url", ""))
    doi = _clean_str(raw.get("doi", "")) or None
    citation_count = max(0, int(raw.get("citation_count") or 0))

    # Enforce field length limits
    title = title[:_MAX_TITLE]
    abstract = abstract[:_MAX_ABSTRACT]
    authors = authors[:_MAX_AUTHORS]

    return Paper(
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        source=source,
        url=url,
        doi=doi,
        citation_count=citation_count,
    )


def normalize_papers(raws: list[dict], source: str) -> list[Paper]:
    """Normalize a batch, silently dropping invalid entries."""
    papers = []
    for raw in raws:
        try:
            p = normalize_paper(raw, source)
            if p:
                papers.append(p)
        except Exception as e:
            log.debug(f"Skipped malformed paper entry from {source}: {e}")
    return papers


def _clean_str(val) -> str:
    if val is None:
        return ""
    return " ".join(str(val).strip().split())


def _clean_authors(val) -> list[str]:
    if not val:
        return []
    if isinstance(val, list):
        return [_clean_str(a) for a in val if _clean_str(a)]
    return [_clean_str(val)]


def _clean_year(val) -> int | None:
    try:
        y = int(val)
        return y if 1900 <= y <= 2100 else None
    except (TypeError, ValueError):
        return None
