"""Token wallet and ledger operations."""

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import TokenLedger, TokenWallet


async def ensure_wallet(db: AsyncSession, user_id: int) -> TokenWallet:
    result = await db.execute(select(TokenWallet).where(TokenWallet.user_id == user_id))
    wallet = result.scalar_one_or_none()
    if wallet:
        return wallet

    wallet = TokenWallet(user_id=user_id, balance=0, version=0)
    db.add(wallet)
    await db.flush()
    await db.refresh(wallet)
    return wallet


async def get_wallet(db: AsyncSession, user_id: int) -> TokenWallet:
    wallet = await ensure_wallet(db, user_id)
    return wallet


async def list_ledger(db: AsyncSession, user_id: int, limit: int = 50, offset: int = 0) -> List[TokenLedger]:
    result = await db.execute(
        select(TokenLedger)
        .where(TokenLedger.user_id == user_id)
        .order_by(TokenLedger.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars().all())


async def credit_tokens(
    db: AsyncSession,
    user_id: int,
    amount: int,
    reason: str,
    reference_type: str,
    reference_id: str,
    idempotency_key: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> TokenLedger:
    if amount <= 0:
        raise ValueError("Credit amount must be positive")

    existing = await db.execute(
        select(TokenLedger).where(
            TokenLedger.user_id == user_id,
            TokenLedger.reason == reason,
            TokenLedger.reference_type == reference_type,
            TokenLedger.reference_id == reference_id,
        )
    )
    found = existing.scalar_one_or_none()
    if found:
        return found

    wallet_stmt = select(TokenWallet).where(TokenWallet.user_id == user_id).with_for_update()
    wallet_result = await db.execute(wallet_stmt)
    wallet = wallet_result.scalar_one_or_none()
    if not wallet:
        wallet = TokenWallet(user_id=user_id, balance=0, version=0)
        db.add(wallet)
        await db.flush()

    before_balance = wallet.balance
    after_balance = before_balance + amount
    wallet.balance = after_balance
    wallet.version += 1

    ledger = TokenLedger(
        wallet_id=wallet.id,
        user_id=user_id,
        direction="credit",
        reason=reason,
        reference_type=reference_type,
        reference_id=str(reference_id),
        delta=amount,
        before_balance=before_balance,
        after_balance=after_balance,
        idempotency_key=idempotency_key,
        metadata_json=metadata,
    )
    db.add(ledger)
    await db.flush()
    await db.refresh(ledger)
    return ledger


async def debit_tokens_once(
    db: AsyncSession,
    user_id: int,
    amount: int,
    reason: str,
    reference_type: str,
    reference_id: str,
    idempotency_key: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[TokenLedger]:
    if amount <= 0:
        raise ValueError("Debit amount must be positive")

    existing = await db.execute(
        select(TokenLedger).where(
            TokenLedger.user_id == user_id,
            TokenLedger.reason == reason,
            TokenLedger.reference_type == reference_type,
            TokenLedger.reference_id == str(reference_id),
        )
    )
    found = existing.scalar_one_or_none()
    if found:
        return found

    wallet_stmt = select(TokenWallet).where(TokenWallet.user_id == user_id).with_for_update()
    wallet_result = await db.execute(wallet_stmt)
    wallet = wallet_result.scalar_one_or_none()
    if not wallet:
        wallet = TokenWallet(user_id=user_id, balance=0, version=0)
        db.add(wallet)
        await db.flush()

    if wallet.balance < amount:
        return None

    before_balance = wallet.balance
    after_balance = before_balance - amount
    wallet.balance = after_balance
    wallet.version += 1

    ledger = TokenLedger(
        wallet_id=wallet.id,
        user_id=user_id,
        direction="debit",
        reason=reason,
        reference_type=reference_type,
        reference_id=str(reference_id),
        delta=amount,
        before_balance=before_balance,
        after_balance=after_balance,
        idempotency_key=idempotency_key,
        metadata_json=metadata,
    )
    db.add(ledger)
    await db.flush()
    await db.refresh(ledger)
    return ledger
