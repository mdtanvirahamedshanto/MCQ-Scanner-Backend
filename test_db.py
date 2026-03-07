import asyncio
from app.database import AsyncSessionLocal
from app.models import Exam
from sqlalchemy import select

async def check():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Exam).order_by(Exam.id.desc()).limit(5))
        exams = result.scalars().all()
        for i, e in enumerate(exams):
            print(f"Exam {e.id}: name={e.exam_name}, template={getattr(e, 'template_type', 'N/A')}")
            
asyncio.run(check())
