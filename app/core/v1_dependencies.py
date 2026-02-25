"""Dependencies for v1 API routes."""

from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_backend_access_token
from app.database import get_db
from app.models import User, UserProfile

security = HTTPBearer()


async def get_current_v1_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    payload = decode_backend_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    sub = payload.get("sub")
    if sub is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    try:
        user_id = int(sub)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


async def get_profile_status(
    user: User,
    db: AsyncSession,
) -> bool:
    result = await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))
    profile = result.scalar_one_or_none()
    if not profile:
        return False
    return bool(profile.institute_name and profile.institute_address)


async def require_profile_complete(
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    profile_complete = await get_profile_status(current_user, db)
    if not profile_complete:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Profile is incomplete. Complete onboarding first.",
        )
    return current_user


async def require_superadmin(
    current_user: User = Depends(get_current_v1_user),
) -> User:
    if current_user.role not in {"superadmin", "admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superadmin access required")
    return current_user


def get_idempotency_key(idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key")) -> str:
    if not idempotency_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Idempotency-Key header is required")
    return idempotency_key
