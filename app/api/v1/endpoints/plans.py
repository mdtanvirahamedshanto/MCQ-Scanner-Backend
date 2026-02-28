"""v1 plan catalog endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.v1_dependencies import get_current_v1_user, require_superadmin
from app.database import get_db
from app.models import Plan, User
from app.schemas_v1 import PlanCreateRequest, PlanResponse, PlanUpdateRequest

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


@router.post("", response_model=PlanResponse)
async def create_plan(
    req: PlanCreateRequest,
    _: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    # Check if code exists
    result = await db.execute(select(Plan).where(Plan.code == req.code))
    if result.first():
        raise HTTPException(status_code=400, detail="Plan with this code already exists")

    new_plan = Plan(
        code=req.code,
        name=req.name,
        price_amount=req.price_amount,
        currency=req.currency,
        billing_cycle=req.billing_cycle,
        tokens_included=req.tokens_included,
        is_active=True,
    )
    db.add(new_plan)
    await db.commit()
    await db.refresh(new_plan)

    return dict(
        id=new_plan.id,
        code=new_plan.code,
        name=new_plan.name,
        price_amount=new_plan.price_amount,
        currency=new_plan.currency,
        billing_cycle=new_plan.billing_cycle,
        tokens_included=new_plan.tokens_included,
        is_active=bool(new_plan.is_active),
    )


@router.put("/{plan_id}", response_model=PlanResponse)
async def update_plan(
    plan_id: int,
    req: PlanUpdateRequest,
    _: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Plan).where(Plan.id == plan_id))
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    if req.code is not None and req.code != plan.code:
        # Check if new code already exists
        code_check = await db.execute(select(Plan).where(Plan.code == req.code, Plan.id != plan_id))
        if code_check.first():
            raise HTTPException(status_code=400, detail="Plan with this code already exists")
        plan.code = req.code

    if req.name is not None:
        plan.name = req.name
    if req.price_amount is not None:
        plan.price_amount = req.price_amount
    if req.currency is not None:
        plan.currency = req.currency
    if req.billing_cycle is not None:
        plan.billing_cycle = req.billing_cycle
    if req.tokens_included is not None:
        plan.tokens_included = req.tokens_included
    if req.is_active is not None:
        plan.is_active = req.is_active

    await db.commit()
    await db.refresh(plan)

    return dict(
        id=plan.id,
        code=plan.code,
        name=plan.name,
        price_amount=plan.price_amount,
        currency=plan.currency,
        billing_cycle=plan.billing_cycle,
        tokens_included=plan.tokens_included,
        is_active=bool(plan.is_active),
    )


@router.delete("/{plan_id}")
async def deactivate_plan(
    plan_id: int,
    _: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Plan).where(Plan.id == plan_id))
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    plan.is_active = False
    await db.commit()
    return {"message": "Plan deactivated successfully"}
