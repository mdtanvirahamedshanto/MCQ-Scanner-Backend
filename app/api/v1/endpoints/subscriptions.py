"""v1 subscription endpoints (manual mode + provider stub)."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.v1_dependencies import get_current_v1_user, require_superadmin
from app.database import get_db
from app.models import Plan, Subscription, User
from app.schemas_v1 import (
    MessageResponse,
    SubscriptionCreateRequest,
    SubscriptionResponse,
    SubscriptionVerifyRequest,
    WebhookStubRequest,
)
from app.services.token_service import credit_tokens

router = APIRouter()


@router.post("", response_model=SubscriptionResponse)
async def create_subscription(
    payload: SubscriptionCreateRequest,
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    if payload.payment_mode != "manual":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only manual payment is enabled currently")

    plan_result = await db.execute(select(Plan).where(Plan.code == payload.plan_code, Plan.is_active == True))
    plan = plan_result.scalar_one_or_none()
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")

    subscription = Subscription(
        user_id=current_user.id,
        plan_id=plan.id,
        status="pending",
        provider="manual",
        provider_ref=payload.transaction_ref,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(subscription)
    await db.flush()
    await db.refresh(subscription)

    return SubscriptionResponse(
        id=subscription.id,
        user_id=subscription.user_id,
        plan_id=subscription.plan_id,
        status=subscription.status,
        provider=subscription.provider,
        provider_ref=subscription.provider_ref,
        starts_at=subscription.starts_at,
        ends_at=subscription.ends_at,
        created_at=subscription.created_at,
    )


@router.get("/me", response_model=Optional[SubscriptionResponse])
async def my_subscription(
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Subscription)
        .where(Subscription.user_id == current_user.id)
        .order_by(Subscription.created_at.desc())
        .limit(1)
    )
    sub = result.scalar_one_or_none()
    if not sub:
        return None
    return SubscriptionResponse(
        id=sub.id,
        user_id=sub.user_id,
        plan_id=sub.plan_id,
        status=sub.status,
        provider=sub.provider,
        provider_ref=sub.provider_ref,
        starts_at=sub.starts_at,
        ends_at=sub.ends_at,
        created_at=sub.created_at,
    )


@router.post("/{subscription_id}/verify", response_model=SubscriptionResponse)
async def verify_subscription(
    subscription_id: int,
    payload: SubscriptionVerifyRequest,
    admin_user: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Subscription).where(Subscription.id == subscription_id))
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found")

    if payload.status == "rejected":
        sub.status = "canceled"
        sub.updated_at = datetime.utcnow()
    else:
        plan_result = await db.execute(select(Plan).where(Plan.id == sub.plan_id))
        plan = plan_result.scalar_one_or_none()
        if not plan:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")

        sub.status = "active"
        sub.starts_at = datetime.utcnow()
        sub.ends_at = datetime.utcnow() + timedelta(days=30)
        sub.activated_by = admin_user.id
        sub.updated_at = datetime.utcnow()

        await credit_tokens(
            db=db,
            user_id=sub.user_id,
            amount=plan.tokens_included,
            reason="subscription_grant",
            reference_type="subscription",
            reference_id=str(sub.id),
            metadata={"plan_code": plan.code, "verified_by": admin_user.id},
        )

    await db.flush()
    return SubscriptionResponse(
        id=sub.id,
        user_id=sub.user_id,
        plan_id=sub.plan_id,
        status=sub.status,
        provider=sub.provider,
        provider_ref=sub.provider_ref,
        starts_at=sub.starts_at,
        ends_at=sub.ends_at,
        created_at=sub.created_at,
    )


@router.post("/webhook", response_model=MessageResponse)
async def subscription_webhook_stub(
    _: WebhookStubRequest,
):
    return MessageResponse(message="Webhook received (stub). Replace with provider adapter later.")
