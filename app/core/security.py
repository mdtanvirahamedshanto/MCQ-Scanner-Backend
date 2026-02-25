"""Security helpers for v1 API."""

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt

from app.config import get_settings

settings = get_settings()


def create_backend_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
    expire_delta = timedelta(minutes=expires_minutes or settings.BACKEND_JWT_EXPIRE_MINUTES)
    exp = datetime.now(timezone.utc) + expire_delta
    payload = {"sub": str(subject), "exp": exp}
    secret = settings.BACKEND_JWT_SECRET or settings.SECRET_KEY
    return jwt.encode(payload, secret, algorithm=settings.ALGORITHM)


def decode_backend_access_token(token: str) -> Optional[dict]:
    candidates = []
    if settings.BACKEND_JWT_SECRET:
        candidates.append(settings.BACKEND_JWT_SECRET)
    if settings.SECRET_KEY and settings.SECRET_KEY not in candidates:
        candidates.append(settings.SECRET_KEY)

    for secret in candidates:
        try:
            return jwt.decode(token, secret, algorithms=[settings.ALGORITHM])
        except JWTError:
            continue
    return None
