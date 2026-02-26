import asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from app.config import get_settings

settings = get_settings()

async def async_main():
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as conn:
        try:
            # Add missing columns to answer_keys
            await conn.execute(sa.text("ALTER TABLE answer_keys ADD COLUMN IF NOT EXISTS set_id INTEGER;"))
            await conn.execute(sa.text("ALTER TABLE answer_keys ADD COLUMN IF NOT EXISTS mapping JSONB;"))
            await conn.execute(sa.text("ALTER TABLE answer_keys ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1;"))
            await conn.execute(sa.text("ALTER TABLE answer_keys ADD COLUMN IF NOT EXISTS is_published BOOLEAN DEFAULT TRUE;"))
            await conn.execute(sa.text("ALTER TABLE answer_keys ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP;"))
            print("Successfully added missing columns to answer_keys.")
            
            # The exam_sets table was likely created by the Base.metadata.create_all() script we ran earlier.
            # But just in case, let's verify.
        except Exception as e:
            print(f"Error: {e}")

asyncio.run(async_main())
