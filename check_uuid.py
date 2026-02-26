import asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from app.config import get_settings
from app.models import Exam, ScanBatch

settings = get_settings()

async def async_main():
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as conn:
        try:
            # Let's search some tables for this string
            res = await conn.execute(sa.text("SELECT * FROM exams LIMIT 5;"))
            print("Exams:", res.fetchall())
            
            res2 = await conn.execute(sa.text("SELECT * FROM scan_batches LIMIT 5;"))
            print("Scan Batches:", res2.fetchall())
            
            res3 = await conn.execute(sa.text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';"))
            print("Tables:", res3.fetchall())
            
        except Exception as e:
            print(f"Error: {e}")

asyncio.run(async_main())
