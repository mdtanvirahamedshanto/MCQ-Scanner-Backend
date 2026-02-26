import asyncio
from app.database import Base, engine
from app.models import *

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("Schema ensured.")

if __name__ == "__main__":
    asyncio.run(main())
