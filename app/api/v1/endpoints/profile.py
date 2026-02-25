"""v1 profile endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.v1_dependencies import get_current_v1_user
from app.database import get_db
from app.models import User, UserProfile
from app.schemas_v1 import ProfileResponse, ProfileStatusResponse, ProfileUpdateRequest

router = APIRouter()


@router.get("", response_model=ProfileResponse)
async def get_profile(
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(UserProfile).where(UserProfile.user_id == current_user.id))
    profile = result.scalar_one_or_none()
    profile_completed = bool(profile and profile.institute_name and profile.institute_address)

    return ProfileResponse(
        institute_name=profile.institute_name if profile else None,
        institute_address=profile.institute_address if profile else None,
        phone=profile.phone if profile else None,
        website=profile.website if profile else None,
        profile_completed=profile_completed,
        profile_completed_at=profile.profile_completed_at if profile else None,
    )


@router.put("", response_model=ProfileResponse)
async def update_profile(
    payload: ProfileUpdateRequest,
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(UserProfile).where(UserProfile.user_id == current_user.id))
    profile = result.scalar_one_or_none()

    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)

    profile.institute_name = payload.institute_name.strip()
    profile.institute_address = payload.institute_address.strip()
    profile.phone = payload.phone.strip() if payload.phone else None
    profile.website = payload.website.strip() if payload.website else None
    profile.profile_completed_at = datetime.utcnow()
    profile.updated_at = datetime.utcnow()

    current_user.institution_name = profile.institute_name
    current_user.address = profile.institute_address

    return ProfileResponse(
        institute_name=profile.institute_name,
        institute_address=profile.institute_address,
        phone=profile.phone,
        website=profile.website,
        profile_completed=True,
        profile_completed_at=profile.profile_completed_at,
    )


@router.get("/status", response_model=ProfileStatusResponse)
async def profile_status(
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(UserProfile).where(UserProfile.user_id == current_user.id))
    profile = result.scalar_one_or_none()
    profile_completed = bool(profile and profile.institute_name and profile.institute_address)
    return ProfileStatusResponse(profile_completed=profile_completed)
