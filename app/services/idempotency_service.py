"""Idempotency helpers."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ScanBatch


async def find_existing_scan_batch(
    db: AsyncSession,
    user_id: int,
    exam_id: int,
    idempotency_key: str,
) -> Optional[ScanBatch]:
    result = await db.execute(
        select(ScanBatch).where(
            ScanBatch.user_id == user_id,
            ScanBatch.exam_id == exam_id,
            ScanBatch.idempotency_key == idempotency_key,
        )
    )
    return result.scalar_one_or_none()
