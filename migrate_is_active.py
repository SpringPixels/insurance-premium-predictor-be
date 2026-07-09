import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/insurance_db")
# Convert asyncpg URL to psycopg2 format
sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

conn = psycopg2.connect(sync_url)
conn.autocommit = True
cur = conn.cursor()

cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;")
print("Migration done: is_active column added (or already existed).")

cur.close()
conn.close()
