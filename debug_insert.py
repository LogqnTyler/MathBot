"""
Standalone debug script — run this to isolate whether inserts to the local
Postgres DB actually commit. Doesn't touch chunk-building logic at all.

Usage (with DATABASE_URL set / in .env):
    python debug_insert.py
"""
import os
import sqlalchemy
from dotenv import load_dotenv

load_dotenv()

database_url = os.environ.get("DATABASE_URL")
print(f"DATABASE_URL = {database_url!r}")

if not database_url:
    raise SystemExit("DATABASE_URL is not set in this shell/session. Set it and retry.")

engine = sqlalchemy.create_engine(database_url)

with engine.connect() as conn:
    print("Connected OK.")
    print("Current database:", conn.execute(sqlalchemy.text("SELECT current_database()")).scalar())
    print("chunks count before:", conn.execute(sqlalchemy.text("SELECT count(*) FROM chunks")).scalar())

# Try a raw insert + explicit commit, completely separate from the ORM/dataclass path
with engine.begin() as conn:  # engine.begin() auto-commits on successful exit, rolls back on exception
    conn.execute(
        sqlalchemy.text("""
            INSERT INTO chunks (kind, mat_id, name, content_plain)
            VALUES ('debug', NULL, 'debug_row_1', 'hello world')
            ON CONFLICT (name, mat_id) DO NOTHING
        """)
    )
    print("Insert executed inside engine.begin() block (should auto-commit on exit).")

with engine.connect() as conn:
    print("chunks count after:", conn.execute(sqlalchemy.text("SELECT count(*) FROM chunks")).scalar())
    rows = conn.execute(sqlalchemy.text("SELECT id, kind, name FROM chunks WHERE name = 'debug_row_1'")).fetchall()
    print("debug row found:", rows)

engine.dispose()