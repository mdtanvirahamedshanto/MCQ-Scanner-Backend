import asyncio
from app.database import AsyncSessionLocal
from sqlalchemy import select
from app.models import AnswerKey

async def main():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AnswerKey).where(AnswerKey.exam_id.in_([7, 9])))
        aks = result.scalars().all()
        for ak in aks:
            print(f"Exam ID: {ak.exam_id}, Set: {ak.set_code}, Number of Answers: {len(ak.answers)}")
            print(f"Answers: {ak.answers}\n")

asyncio.run(main())
