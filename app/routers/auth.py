"""Authentication endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User
from app.schemas import UserCreate, UserResponse, Token, LoginRequest
from app.auth import get_password_hash, verify_password, create_access_token
from app.config import get_settings
from app.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
@router.post("/signup", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new teacher/user."""
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    settings = get_settings()
    role = "admin" if settings.ADMIN_EMAIL and user_data.email.lower() == settings.ADMIN_EMAIL.lower() else "teacher"
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        role=role,
        institution_name=user_data.institution_name,
        address=user_data.address,
        tokens=500,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current user info (for role check)."""
    updated = False
    if current_user.role is None:
        current_user.role = "teacher"
        updated = True
    if current_user.is_subscribed is None:
        current_user.is_subscribed = False
        updated = True
    if current_user.tokens is None:
        current_user.tokens = 500
        updated = True

    if updated:
        db.add(current_user)
        await db.flush()
        await db.refresh(current_user)

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        role=current_user.role or "teacher",
        is_subscribed=bool(current_user.is_subscribed),
        subscription_plan=current_user.subscription_plan,
        institution_name=current_user.institution_name,
        address=current_user.address,
        tokens=current_user.tokens if current_user.tokens is not None else 500,
        created_at=current_user.created_at or datetime.utcnow(),
    )


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Login and receive JWT access token."""
    result = await db.execute(select(User).where(User.email == login_data.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    access_token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=access_token, role=user.role)
