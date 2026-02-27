"""
schemas/response.py — All agent response formatters live here.
Agents return data objects. This module converts them to user-readable markdown.
Agents import from here; they never build strings inline.
"""
from memory.models import Paper, Dataset, Devlog, LinkedInPost


# ─── Research ─────────────────────────────────────────────────────────────────

def format_research(papers: list[Paper], synthesis: dict, model: str) -> str:
    lines = [
        "## 🔬 Research Results",
        f"*{len(papers)} papers · ArXiv + Semantic Scholar*",
        "",
        "### Synthesis",
        synthesis.get("synthesis", ""),
        "",
        "### Papers Found",
    ]
    for i, p in enumerate(papers[:6], 1):
        authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
        cite = f" · {p.citation_count:,} citations" if p.citation_count else ""
        lines.append(f"**{i}. [{p.title}]({p.url})** ({p.year or '?'}) — {authors}{cite}")
    lines.append("")

    if synthesis.get("key_insights"):
        lines += ["### Key Insights", *[f"- {i}" for i in synthesis["key_insights"]], ""]

    datasets = synthesis.get("datasets_mentioned", [])
    if datasets:
        lines += [
            "### Datasets Mentioned",
            f"*{', '.join(datasets)}*",
            "> 💡 Say **'find datasets'** to search for these automatically.",
            "",
        ]

    if synthesis.get("tags"):
        lines.append(f"*Tags: {', '.join(synthesis['tags'])}*")

    lines.append(f"\n*via {model}*")
    return "\n".join(lines)


def format_research_no_results(topic: str, fallback_content: str, model: str) -> str:
    return (
        f"## 🔬 Research: {topic}\n\n"
        f"*Note: Live API search returned no results — using training knowledge.*\n\n"
        f"{fallback_content}\n\n*via {model}*"
    )


# ─── Dataset ──────────────────────────────────────────────────────────────────

def format_datasets(ranked: list[dict], recommendation: str, topic: str, model: str) -> str:
    icons = {"huggingface": "🤗", "kaggle": "🏆"}
    lines = [
        f"## 📊 Top Datasets for: {topic}",
        f"*Searched HuggingFace + Kaggle · Ranked by relevance, popularity, usability*",
        "",
    ]
    for i, ds in enumerate(ranked, 1):
        icon = icons.get(ds.get("source", ""), "📦")
        url = ds.get("url", "")
        score = ds.get("score", "?")
        lines.append(f"### {i}. {icon} [{ds.get('name', '?')}]({url}) — Score: {score}/100")
        if ds.get("size"):
            lines.append(f"**Size:** {ds['size']}  |  **Downloads:** {ds.get('downloads', 0):,}")
        lines.append(f"**Why it fits:** {ds.get('relevance_reason', '')}")
        lines.append(f"**Use for:** {ds.get('use_for', '')}")
        if ds.get("limitations"):
            lines.append(f"**Watch out for:** {ds['limitations']}")
        lines.append("")

    if recommendation:
        lines += ["### 💡 Recommendation", recommendation, ""]

    lines.append(f"*Results cached in memory · via {model}*")
    return "\n".join(lines)


def format_datasets_empty(topic: str) -> str:
    return (
        f"⚠️ No datasets found for **'{topic}'**.\n\n"
        "Try:\n"
        "- Broader search terms (e.g. 'sentiment' instead of 'movie review sentiment')\n"
        "- Adding `KAGGLE_USERNAME` and `KAGGLE_KEY` to `.env` for Kaggle results\n"
        "- Searching HuggingFace directly at [hf.co/datasets](https://huggingface.co/datasets)"
    )


# ─── GitHub / Devlog ──────────────────────────────────────────────────────────

def format_devlog(devlog: Devlog, commit_result: dict | None, dry_run: bool, model: str) -> str:
    STATUS_ICONS = {"done": "✅", "in_progress": "🔄", "blocked": "🚫", "planned": "📋"}
    lines = [
        f"## 📝 Devlog — {devlog.date.strftime('%B %d, %Y')}",
        "",
        devlog.summary,
        "",
        "### Tasks",
    ]
    for t in devlog.tasks:
        note = f" — *{t.notes}*" if t.notes else ""
        lines.append(f"- {STATUS_ICONS.get(t.status, '•')} {t.task}{note}")

    if devlog.topics:
        lines += ["", f"**Topics:** {', '.join(devlog.topics)}"]
    if devlog.mood:
        lines.append(f"**Mood:** {'⭐' * devlog.mood}")

    lines.append("")

    if dry_run and commit_result:
        target = commit_result.get("would_commit_to", "")
        lines.append(f"🧪 **Dry Run** — would commit to `{target}`")
        lines.append("*Set `GITHUB_DRY_RUN=false` in `.env` to enable real commits.*")
    elif commit_result and commit_result.get("skipped"):
        lines.append(f"⏭️ **Commit skipped** — {commit_result.get('reason')}")
    elif commit_result and commit_result.get("sha"):
        sha, url, branch = commit_result["sha"], commit_result["url"], commit_result["branch"]
        lines.append(f"✅ **Committed** → [`{sha}`]({url}) on `{branch}`")
    elif not dry_run:
        lines.append("💾 **Saved locally.** Add `GITHUB_TOKEN` + `GITHUB_REPO` to enable commits.")

    lines.append(f"\n*via {model}*")
    return "\n".join(lines)


# ─── LinkedIn ──────────────────────────────────────────────────────────────────

def format_linkedin_draft(post: LinkedInPost, model: str) -> str:
    post_id = str(post.id)[:8] if post.id else "new"
    return (
        f"## 💼 LinkedIn Post — `{post.status.upper()}`\n"
        f"*{post.word_count} words · ID: `{post_id}...`*\n\n"
        f"---\n\n"
        f"{post.content}\n\n"
        f"---\n\n"
        f"**Status:** `DRAFT` — copy above and paste into LinkedIn to publish.\n"
        f"**Adjust tone:** *'set my LinkedIn tone to technical'* or *'shorter posts'*\n\n"
        f"*via {model}*"
    )


def format_linkedin_no_content() -> str:
    return (
        "⚠️ **No content found in memory.**\n\n"
        "Jarvis needs something to write about. Try:\n"
        "- *'Record today: built X, fixed Y'* — to log progress\n"
        "- *'Find papers on Z'* — to log research"
    )
