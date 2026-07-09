import sqlite3

try:
    conn = sqlite3.connect('health_guard.db')
    cursor = conn.cursor()
    cursor.execute("ALTER TABLE activity_logs ADD COLUMN is_recommended BOOLEAN DEFAULT 0;")
    conn.commit()
    print("Column added successfully.")
except sqlite3.OperationalError as e:
    print(f"OperationalError: {e}")
except Exception as e:
    print(f"Error: {e}")
finally:
    if conn:
        conn.close()
