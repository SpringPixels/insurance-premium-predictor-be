import asyncio
from sqlalchemy import text
from config.database import engine

async def migrate():
    async with engine.begin() as conn:
        try:
            await conn.execute(text("ALTER TABLE activity_logs ADD COLUMN is_recommended BOOLEAN DEFAULT FALSE;"))
            print("Column added successfully.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(migrate())
