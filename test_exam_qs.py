import asyncio
from app.database import AsyncSessionLocal
from sqlalchemy import select
from app.models import Exam, AnswerKey

async def main():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Exam))
        exams = result.scalars().all()
        for e in exams:
            print(f"Exam ID: {e.id}, Title: {e.title}, Total Qs: {e.total_questions}")
        
        result2 = await session.execute(select(AnswerKey).where(AnswerKey.exam_id == 7))
        aks = result2.scalars().all()
        for ak in aks:
            print(f"Exam 7 AK length: {len(ak.answers)}")

asyncio.run(main())
