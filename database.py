import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

# SQLite async URL
DATABASE_URL = "postgresql+asyncpg://postgres:12345678@localhost:5432/postgres"

# Asynchronous engine create kar rahe hain
engine = create_async_engine(DATABASE_URL, echo=True)

# Async session maker
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


# Dependency to get async DB session
async def get_db():
  async with AsyncSessionLocal() as session:
    yield session