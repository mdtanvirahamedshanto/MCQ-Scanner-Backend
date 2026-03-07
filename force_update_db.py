import asyncio
import sys
from app.database import AsyncSessionLocal
from app.models import Exam
from sqlalchemy import select, update

async def force_update():
    async with AsyncSessionLocal() as session:
        # Update all exams to use the new template to save the user from recreating them
        await session.execute(
            update(Exam).values(template_type="20q_mcq_png")
        )
        await session.commit()
        print("Successfully updated all exams to '20q_mcq_png'!")
            
if __name__ == "__main__":
    asyncio.run(force_update())
