"""
GitHub Agent v3 — devlogs with dry run, diff check, branch isolation, schemas format.
"""
import base64
import httpx
from datetime import date
from agents.base import BaseAgent
from core.config import settings
from core.exceptions import AgentError, ProviderError
from core.safety import check_permission, Permission, PermissionDenied
from memory.models import Devlog, Task, CommitMetadata
from memory.service import sanitize_for_llm
from router.llm_router import LLMRouter
from schemas.api import DevlogParse
from schemas.response import format_devlog
from core.logger import get_logger

log = get_logger(__name__)

PARSE_PROMPT = """Parse the developer's progress description into a structured devlog.

Respond in JSON only:
{
  "summary": "<1-2 paragraph narrative>",
  "tasks": [{"task": "...", "status": "done|in_progress|blocked|planned", "notes": "optional"}],
  "topics": ["topic1"],
  "mood": <1-5 or omit>
}
"""


class GitHubAgent(BaseAgent):
    name = "github"
    description = "Creates devlogs, commits to jarvis/devlog branch (never main)"

    async def run(self, user_input: str, session_id: str, user_id: str, **kwargs) -> str:
        start_time = self.logger.start(session_id, user_input)
        today = date.today()
        dry_run = settings.GITHUB_DRY_RUN

        if dry_run:
            self.logger.info("DRY RUN active — no real commits")

        # Parse progress
        existing = self.memory.get_devlog(user_id, today)
        extra = f"\n\nExisting entry to update:\n{existing.summary}" if existing else ""

        response = await self.llm(
            system=PARSE_PROMPT,
            task_type="classification",
            messages=[{"role": "user", "content": f"Progress:\n{sanitize_for_llm(user_input)}{extra}"}],
            max_tokens=800, temperature=0.3, json_mode=True,
        )

        try:
            raw = LLMRouter.extract_json(response.content)
            parsed = DevlogParse.from_llm(raw)
        except (ValueError, Exception) as e:
            raise AgentError(self.name, f"Could not parse progress into devlog structure: {e}")

        devlog = Devlog(
            user_id=user_id,
            date=today,
            summary=parsed.summary or user_input,
            tasks=[Task(**t) for t in parsed.tasks],
            topics=parsed.topics,
            mood=parsed.mood,
        )
        devlog = self.memory.upsert_devlog(devlog)

        # Commit to GitHub
        commit_result = None
        if dry_run:
            commit_result = {"dry_run": True, "would_commit_to": f"{settings.GITHUB_DEVLOG_BRANCH}/docs/devlog/{today}.md"}
        elif settings.GITHUB_TOKEN and settings.GITHUB_REPO:
            try:
                check_permission(self.name, Permission.GITHUB_WRITE)
                commit_result = await self._commit(devlog)
            except (PermissionDenied, ProviderError) as e:
                self.logger.warn(f"GitHub commit blocked: {e}")
                commit_result = None

        self.logger.success(start_time, response.model_used, f"devlog {today}")
        out = format_devlog(devlog, commit_result, dry_run, response.model_used)
        self._save_agent_message(session_id, out)
        return out

    async def _commit(self, devlog: Devlog) -> dict | None:
        path = f"docs/devlog/{devlog.date.isoformat()}.md"
        new_md = devlog.to_markdown()
        encoded = base64.b64encode(new_md.encode()).decode()
        branch = settings.GITHUB_DEVLOG_BRANCH
        url = f"https://api.github.com/repos/{settings.GITHUB_REPO}/contents/{path}"
        headers = {
            "Authorization": f"token {settings.GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }

        async with httpx.AsyncClient(timeout=20) as client:
            await self._ensure_branch(client, headers, branch)

            sha, old_md = None, ""
            check = await client.get(url, headers=headers, params={"ref": branch})
            if check.status_code == 200:
                file_data = check.json()
                sha = file_data.get("sha")
                try:
                    old_md = base64.b64decode(file_data["content"].replace("\n", "")).decode()
                except Exception:
                    pass

            diff = len(new_md) - len(old_md)
            if sha and abs(diff) < settings.GITHUB_MIN_DIFF_CHARS:
                self.logger.skip(f"diff only {diff} chars — below threshold")
                return {"skipped": True, "reason": f"diff {diff} chars < {settings.GITHUB_MIN_DIFF_CHARS} threshold"}

            payload = {
                "message": f"devlog({devlog.date}): {', '.join(devlog.topics[:3]) or 'progress'}",
                "content": encoded,
                "branch": branch,
            }
            if sha:
                payload["sha"] = sha

            resp = await client.put(url, headers=headers, json=payload)

        if resp.status_code not in (200, 201):
            raise ProviderError("github", f"Commit failed with HTTP {resp.status_code}", resp.status_code)

        sha7 = resp.json().get("commit", {}).get("sha", "")[:7]
        commit_url = f"https://github.com/{settings.GITHUB_REPO}/blob/{branch}/{path}"

        meta = CommitMetadata(
            commit_sha=sha7, summary=devlog.summary[:200],
            branch=branch, repo=settings.GITHUB_REPO, url=commit_url,
        )
        devlog.commits.append(sha7)
        devlog.commit_metadata.append(meta)
        devlog.github_path = path
        self.memory.upsert_devlog(devlog)

        self.logger.info(f"Committed {sha7} → {branch}")
        return {"sha": sha7, "url": commit_url, "branch": branch}

    async def _ensure_branch(self, client, headers, branch):
        main = await client.get(
            f"https://api.github.com/repos/{settings.GITHUB_REPO}/git/refs/heads/main",
            headers=headers
        )
        if main.status_code != 200:
            return
        main_sha = main.json().get("object", {}).get("sha")
        if not main_sha:
            return
        check = await client.get(
            f"https://api.github.com/repos/{settings.GITHUB_REPO}/git/refs/heads/{branch}",
            headers=headers
        )
        if check.status_code == 200:
            return
        await client.post(
            f"https://api.github.com/repos/{settings.GITHUB_REPO}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{branch}", "sha": main_sha},
        )
        self.logger.info(f"Created branch: {branch}")


github_agent = GitHubAgent()
