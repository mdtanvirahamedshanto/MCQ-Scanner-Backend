"""Redis-backed lightweight rate-limiting helpers."""

from __future__ import annotations

import time

import redis
from fastapi import HTTPException, status

from app.config import get_settings

settings = get_settings()


def _get_client() -> redis.Redis:
    return redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def enforce_rate_limit(key: str, limit: int, window_seconds: int) -> None:
    """Raise 429 when key exceeds limit inside time window."""
    if limit <= 0:
        return

    now = int(time.time())
    window_key = f"rl:{key}:{now // window_seconds}"

    try:
        client = _get_client()
        count = client.incr(window_key)
        if count == 1:
            client.expire(window_key, window_seconds)
        if count > limit:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    except redis.RedisError:
        # fail-open in local/dev if redis is unavailable
        return
