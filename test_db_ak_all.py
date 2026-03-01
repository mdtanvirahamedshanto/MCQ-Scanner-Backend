import asyncio
from app.database import AsyncSessionLocal
from sqlalchemy import select
from app.models import AnswerKey

async def main():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AnswerKey))
        aks = result.scalars().all()
        print(f"Total answer keys in DB: {len(aks)}")
        for ak in aks:
            print(f"Exam ID: {ak.exam_id}, Set: {ak.set_code}, Number of Answers: {len(ak.answers)}")

asyncio.run(main())
