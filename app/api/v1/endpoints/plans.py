"""v1 plan catalog endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.v1_dependencies import get_current_v1_user
from app.database import get_db
from app.models import Plan, User
from app.schemas_v1 import PlanResponse

router = APIRouter()


DEFAULT_PLANS = [
    {"code": "BASIC_500", "name": "Basic", "price_amount": 500, "currency": "BDT", "billing_cycle": "monthly", "tokens_included": 500},
    {"code": "PRO_2000", "name": "Pro", "price_amount": 1500, "currency": "BDT", "billing_cycle": "monthly", "tokens_included": 2000},
    {"code": "INSTITUTE_10000", "name": "Institute", "price_amount": 5000, "currency": "BDT", "billing_cycle": "monthly", "tokens_included": 10000},
]


async def _seed_default_plans_if_needed(db: AsyncSession) -> None:
    result = await db.execute(select(Plan.id))
    has_any = result.first() is not None
    if has_any:
        return

    for item in DEFAULT_PLANS:
        db.add(Plan(**item, is_active=True))
    await db.flush()


@router.get("", response_model=list[PlanResponse])
async def list_plans(
    _: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    await _seed_default_plans_if_needed(db)
    result = await db.execute(select(Plan).where(Plan.is_active == True).order_by(Plan.id.asc()))
    plans = result.scalars().all()
    return [
        PlanResponse(
            id=p.id,
            code=p.code,
            name=p.name,
            price_amount=p.price_amount,
            currency=p.currency,
            billing_cycle=p.billing_cycle,
            tokens_included=p.tokens_included,
            is_active=bool(p.is_active),
        )
        for p in plans
    ]
