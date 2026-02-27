"""
api/auth.py — Authentication endpoints.

POST /api/auth/token   → issue JWT (dev mode: any user_id accepted)
GET  /api/auth/me      → validate token, return user_id
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from core.auth import create_token, decode_token, AuthError, get_current_user
from core.config import settings
from core.logger import get_logger
from schemas.api import JarvisResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])
log = get_logger("api.auth")


class TokenRequest(BaseModel):
    user_id: str
    # In production: replace with real credential validation
    # (Supabase email/password, OAuth token exchange, etc.)
    # For MVP: any user_id issues a valid token
    secret: str = ""   # optional dev passphrase guard


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    expires_in_hours: int = 24


@router.post("/token", response_model=JarvisResponse)
async def issue_token(req: TokenRequest):
    """
    Issue a JWT for a user_id.

    MVP mode (AUTH_REQUIRED=false): open to anyone — good for local dev.
    Production (AUTH_REQUIRED=true): validate req.secret against AUTH_TOKEN_SECRET.
    """
    if not req.user_id or len(req.user_id) < 3:
        raise HTTPException(status_code=400, detail="user_id must be at least 3 characters")

    # Production gate: if a token secret is configured, require it
    if settings.AUTH_REQUIRED and settings.AUTH_TOKEN_SECRET:
        if req.secret != settings.AUTH_TOKEN_SECRET:
            log.warning(f"Token issue rejected for user_id={req.user_id[:8]} (bad secret)")
            raise HTTPException(status_code=403, detail="Invalid secret")

    try:
        token = create_token(req.user_id)
        log.info(f"Token issued for user_id={req.user_id[:8]}")
        return JarvisResponse.success(data=TokenResponse(
            access_token=token,
            user_id=req.user_id,
        ))
    except AuthError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me", response_model=JarvisResponse)
async def get_me(user_id: str = Depends(get_current_user)):
    """Validate token and return current user identity."""
    return JarvisResponse.success(data={"user_id": user_id, "authenticated": True})
