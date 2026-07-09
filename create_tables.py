import asyncio
from config.database import engine
from ml_model.db_models import Base

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("Tables created successfully.")

if __name__ == "__main__":
    asyncio.run(init_models())
