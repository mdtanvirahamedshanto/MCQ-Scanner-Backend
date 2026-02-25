"""v1 wallet and ledger endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.v1_dependencies import get_current_v1_user
from app.database import get_db
from app.models import User
from app.schemas_v1 import WalletLedgerEntryResponse, WalletLedgerListResponse, WalletResponse
from app.services.token_service import get_wallet, list_ledger

router = APIRouter()


@router.get("", response_model=WalletResponse)
async def wallet_balance(
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    wallet = await get_wallet(db, current_user.id)
    return WalletResponse(balance=wallet.balance)


@router.get("/ledger", response_model=WalletLedgerListResponse)
async def wallet_ledger(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    wallet = await get_wallet(db, current_user.id)
    rows = await list_ledger(db, current_user.id, limit=limit, offset=offset)
    entries = [
        WalletLedgerEntryResponse(
            id=r.id,
            direction=r.direction,
            reason=r.reason,
            reference_type=r.reference_type,
            reference_id=r.reference_id,
            delta=r.delta,
            before_balance=r.before_balance,
            after_balance=r.after_balance,
            created_at=r.created_at,
        )
        for r in rows
    ]
    return WalletLedgerListResponse(balance=wallet.balance, entries=entries)
