import asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from app.config import get_settings

settings = get_settings()

async def async_main():
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as conn:
        try:
            await conn.execute(sa.text("ALTER TABLE exams ADD COLUMN IF NOT EXISTS exam_name VARCHAR(255);"))
            await conn.execute(sa.text("ALTER TABLE exams ADD COLUMN IF NOT EXISTS subject_name VARCHAR(255);"))
            await conn.execute(sa.text("ALTER TABLE exams ADD COLUMN IF NOT EXISTS has_set BOOLEAN DEFAULT FALSE;"))
            await conn.execute(sa.text("ALTER TABLE exams ADD COLUMN IF NOT EXISTS set_count INTEGER DEFAULT 1;"))
            await conn.execute(sa.text("ALTER TABLE exams ADD COLUMN IF NOT EXISTS options_per_question INTEGER DEFAULT 4;"))
            await conn.execute(sa.text("ALTER TABLE exams ADD COLUMN IF NOT EXISTS negative_marking BOOLEAN DEFAULT FALSE;"))
            await conn.execute(sa.text("ALTER TABLE exams ADD COLUMN IF NOT EXISTS negative_value FLOAT DEFAULT 0.0;"))
            await conn.execute(sa.text("ALTER TABLE exams ADD COLUMN IF NOT EXISTS mark_per_question FLOAT DEFAULT 1.0;"))
            await conn.execute(sa.text("ALTER TABLE exams ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'draft';"))
            await conn.execute(sa.text("ALTER TABLE exams ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP;"))
            print("Successfully added missing columns to exams.")
        except Exception as e:
            print(f"Error: {e}")

asyncio.run(async_main())
