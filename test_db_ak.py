import asyncio
from app.database import AsyncSessionLocal
from sqlalchemy import select
from app.models import AnswerKey

async def main():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AnswerKey).limit(1))
        ak = result.scalar_one_or_none()
        if ak:
            print(f"Set Code: {ak.set_code}")
            print(f"Answers: {ak.answers}")
        else:
            print("No AnswerKey found")

asyncio.run(main())
