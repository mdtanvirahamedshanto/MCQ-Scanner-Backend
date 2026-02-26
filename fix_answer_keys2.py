import asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from app.config import get_settings

settings = get_settings()

async def async_main():
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as conn:
        try:
            # Add missing created_at to answer_keys
            await conn.execute(sa.text("ALTER TABLE answer_keys ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"))
            print("Successfully added created_at to answer_keys.")
            # Verify exam_sets table exists
            res = await conn.execute(sa.text("SELECT table_name FROM information_schema.tables WHERE table_name='exam_sets';"))
            if not res.fetchone():
                print("WARNING: exam_sets does not exist!")
        except Exception as e:
            print(f"Error: {e}")

asyncio.run(async_main())
