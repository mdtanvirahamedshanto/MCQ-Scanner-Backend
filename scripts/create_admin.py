#!/usr/bin/env python3
"""
Create an admin user for OptiMark.
Usage:
    python scripts/create_admin.py admin@example.com mypassword
    python scripts/create_admin.py  # interactive mode
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from app.database import AsyncSessionLocal, engine
from app.models import User
from app.auth import get_password_hash


async def create_admin(email: str, password: str) -> bool:
    """Create or update user to admin role."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if user:
            user.role = "admin"
            user.hashed_password = get_password_hash(password)
            await db.commit()
            print(f"✓ Updated existing user '{email}' to admin role")
        else:
            user = User(
                email=email,
                hashed_password=get_password_hash(password),
                role="admin",
            )
            db.add(user)
            await db.commit()
            print(f"✓ Created new admin user: {email}")
        return True


async def _main():
    try:
        if len(sys.argv) >= 3:
            await create_admin(sys.argv[1], sys.argv[2])
        else:
            email = input("Admin email: ").strip()
            password = input("Admin password: ").strip()
            if not email or not password:
                print("Error: email and password required")
                sys.exit(1)
            await create_admin(email, password)
    finally:
        await engine.dispose()


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
