"""
memory/compactor.py — Long-term memory management for Jarvis.

STATUS: Stub. Core compression logic lives in MemoryService.compress_old_memories().
This module is the scheduled runner for that logic.

CURRENT STATE:
  compress_old_memories() exists in service.py and works correctly.
  It is exposed via POST /api/memory/compress for manual triggering.

WHEN TO ACTIVATE THIS:
  When daily active usage makes the knowledge_entries table grow beyond ~5,000 rows,
  or when search quality degrades because old irrelevant entries pollute results.

DESIGN (implement when needed):
  1. Run daily via a background scheduler (APScheduler or Railway cron)
  2. Compress entries older than 30 days with importance_score < 0.7
  3. Group by topic tag, summarize each group, delete originals
  4. Log compression stats to Supabase for monitoring

FUTURE IMPLEMENTATION SKETCH:

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from memory.service import memory_service
    from core.config import settings
    from core.logger import get_logger

    log = get_logger("memory.compactor")

    scheduler = AsyncIOScheduler()

    @scheduler.scheduled_job("cron", hour=2)  # 2 AM daily
    async def nightly_compress():
        log.info("Nightly memory compression started")
        compressed = await memory_service.compress_old_memories(
            user_id=settings.DEFAULT_USER_ID,
            older_than_days=30,
        )
        log.info(f"Nightly compression complete: {compressed} entries compressed")

    def start():
        scheduler.start()
        log.info("Memory compactor scheduler started")

MANUAL TRIGGER (available now):
  POST /api/memory/compress
  Query params: user_id, older_than_days (default 30)

DEPENDENCIES TO ADD WHEN ACTIVATING:
  apscheduler==3.10.4  # add to requirements.txt
"""

# Intentionally not runnable — see docstring above.
# Activate by implementing the sketch above and calling start() from main.py.
