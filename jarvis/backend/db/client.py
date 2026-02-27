from functools import lru_cache
from supabase import create_client, Client
from core.config import settings
from core.exceptions import ConfigError
from core.logger import db_log


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        raise ConfigError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    db_log.debug("Supabase client initialized")
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
