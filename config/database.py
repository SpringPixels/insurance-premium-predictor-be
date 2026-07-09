from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:7439466632@db:5432/insurance_db")

engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = async_sessionmaker(expire_on_commit=False, bind=engine)
Base = declarative_base()

# Change your get_db() function to match this async layout
async def get_db():
    async with SessionLocal() as db:
        try:
            yield db
        finally:
            await db.close() # Notice the 'await' here