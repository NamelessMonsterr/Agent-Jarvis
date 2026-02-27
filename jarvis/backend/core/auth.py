"""
core/auth.py — JWT authentication for Jarvis.

Design:
  - Stateless JWT tokens (no DB round-trip per request)
  - Compatible with Supabase Auth JWTs (same secret, same algorithm)
  - Gradual rollout: AUTH_REQUIRED=false lets you deploy first, enable auth later
  - Every token carries user_id — memory and actions are scoped to it

Token flow:
  POST /api/auth/token  → issue JWT
  All other endpoints   → Bearer token in Authorization header
  Middleware extracts   → user_id injected into request state
"""
import time
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.config import settings
from core.exceptions import JarvisError
from core.logger import get_logger

log = get_logger("auth")

# ─── JWT (minimal, no extra dependency if python-jose not installed) ──────────
# Using PyJWT which is already a transitive dependency of supabase-py.
try:
    import jwt as pyjwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False
    log.warning("PyJWT not installed — AUTH_REQUIRED must be false. Run: pip install PyJWT")


class AuthError(JarvisError):
    pass


_bearer = HTTPBearer(auto_error=False)


# ─── Token creation ───────────────────────────────────────────────────────────

def create_token(user_id: str, expires_in_hours: int = 24) -> str:
    """Issue a signed JWT for a user."""
    if not _JWT_AVAILABLE:
        raise AuthError("PyJWT not installed")
    if not settings.JWT_SECRET:
        raise AuthError("JWT_SECRET not configured in .env")

    payload = {
        "sub":  user_id,
        "iat":  int(time.time()),
        "exp":  int(time.time()) + expires_in_hours * 3600,
        "jti":  secrets.token_hex(8),   # unique token ID
    }
    return pyjwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    """Decode and validate a JWT. Raises AuthError on any failure."""
    if not _JWT_AVAILABLE:
        raise AuthError("PyJWT not installed")
    if not settings.JWT_SECRET:
        raise AuthError("JWT_SECRET not configured")
    try:
        return pyjwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except pyjwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except pyjwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {e}")


# ─── FastAPI dependency ───────────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    """
    FastAPI dependency. Returns user_id from token.

    Behavior:
      - AUTH_REQUIRED=true  → token mandatory, raises 401 if missing/invalid
      - AUTH_REQUIRED=false → token optional; falls back to DEFAULT_USER_ID
        (safe for local dev and first deployment)
    """
    if not settings.AUTH_REQUIRED:
        # Auth disabled — use token if provided, otherwise use default
        if credentials and credentials.credentials:
            try:
                payload = decode_token(credentials.credentials)
                return payload["sub"]
            except AuthError:
                pass   # Invalid token in dev mode → still allow with default
        return settings.DEFAULT_USER_ID

    # Auth required
    if not credentials or not credentials.credentials:
        log.warning(f"Unauthenticated request to {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Include 'Authorization: Bearer <token>' header.",
        )

    try:
        payload = decode_token(credentials.credentials)
        user_id = payload.get("sub", "")
        if not user_id:
            raise AuthError("Token missing user ID (sub claim)")
        log.debug(f"Authenticated user: {user_id[:8]}")
        return user_id
    except AuthError as e:
        log.warning(f"Auth failed for {request.url.path}: {e}")
        raise HTTPException(status_code=401, detail=str(e))
