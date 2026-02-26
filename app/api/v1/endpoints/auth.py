"""v1 auth endpoints (Google exchange + session)."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.security import create_backend_access_token
from app.core.v1_dependencies import get_current_v1_user
from app.database import get_db
from app.models import TokenWallet, User, UserProfile
from app.schemas_v1 import AuthExchangeResponse, AuthSessionResponse, AuthUserResponse, GoogleExchangeRequest

router = APIRouter()
settings = get_settings()


def _safe_generated_password(sub: str) -> str:
    return f"oauth-google::{sub}"


@router.post("/oauth/google/exchange", response_model=AuthExchangeResponse)
async def google_exchange(
    payload: GoogleExchangeRequest,
    db: AsyncSession = Depends(get_db),
):
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth is not configured on the server",
        )

    try:
        id_info = google_id_token.verify_oauth2_token(
            payload.id_token,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google token") from exc

    google_sub = id_info.get("sub")
    email = (id_info.get("email") or "").lower().strip()
    if not google_sub or not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google token missing required fields")

    query = await db.execute(select(User).where((User.google_sub == google_sub) | (User.email == email)))
    user = query.scalar_one_or_none()

    if user is None:
        user = User(
            email=email,
            google_sub=google_sub,
            hashed_password=_safe_generated_password(google_sub),
            role="teacher",
            is_active=True,
            is_subscribed=False,
            tokens=0,
            created_at=datetime.utcnow(),
        )
        db.add(user)
        await db.flush()
    else:
        if not user.google_sub:
            user.google_sub = google_sub
        if not user.is_active:
            user.is_active = True

    profile_result = await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))
    profile = profile_result.scalar_one_or_none()
    if not profile:
        db.add(UserProfile(user_id=user.id))

    wallet_result = await db.execute(select(TokenWallet).where(TokenWallet.user_id == user.id))
    wallet = wallet_result.scalar_one_or_none()
    if not wallet:
        db.add(TokenWallet(user_id=user.id, balance=0, version=0))

    access_token = create_backend_access_token(str(user.id))

    profile_completed = bool(
        profile and profile.institute_name and profile.institute_address
    )

    return AuthExchangeResponse(
        access_token=access_token,
        user=AuthUserResponse(
            id=user.id,
            email=user.email,
            role=user.role or "teacher",
            profile_completed=profile_completed,
        ),
    )


@router.get("/session", response_model=AuthSessionResponse)
async def session_check(
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    profile_result = await db.execute(select(UserProfile).where(UserProfile.user_id == current_user.id))
    profile = profile_result.scalar_one_or_none()

    # Safety net: auto-create UserProfile for legacy-registered users
    if not profile:
        profile = UserProfile(
            user_id=current_user.id,
            institute_name=current_user.institution_name,
            institute_address=current_user.address,
        )
        if current_user.institution_name and current_user.address:
            profile.profile_completed_at = datetime.utcnow()
        db.add(profile)
        await db.flush()

    # Safety net: auto-create TokenWallet for legacy-registered users
    wallet_result = await db.execute(select(TokenWallet).where(TokenWallet.user_id == current_user.id))
    if not wallet_result.scalar_one_or_none():
        db.add(TokenWallet(user_id=current_user.id, balance=0, version=0))
        await db.flush()

    profile_completed = bool(profile and profile.institute_name and profile.institute_address)

    return AuthSessionResponse(
        authenticated=True,
        user=AuthUserResponse(
            id=current_user.id,
            email=current_user.email,
            role=current_user.role or "teacher",
            profile_completed=profile_completed,
        ),
    )
