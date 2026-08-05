import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
import pg8000
import sqlalchemy

load_dotenv()

connector = Connector(refresh_strategy="LAZY")

database_url = os.environ.get("DATABASE_URL")

if database_url:

    def create_engine():
        return sqlalchemy.create_engine(database_url)

else:

    def create_engine() -> sqlalchemy.engine.base.Engine:
        """
        Initializes a connection pool for a Cloud SQL instance of Postgres.

        Uses the Cloud SQL Python Connector package.
        """
        instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]
        db_user = os.environ["DB_USER"]
        db_pass = os.environ["DB_PASS"]
        db_name = os.environ["DB_NAME"]

        ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

        def getconn() -> pg8000.dbapi.Connection:
            conn: pg8000.dbapi.Connection = connector.connect(
                instance_connection_name,
                "pg8000",
                user=db_user,
                password=db_pass,
                db=db_name,
                ip_type=ip_type,
            )
            return conn

        return sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
        )


def main() -> None:
    json_dir = Path(__file__).resolve().parents[1] / "JSON"
    lesson_files = sorted(json_dir.glob("lesson*.json"))

    if not lesson_files:
        print(f"No lesson*.json files found in {json_dir}")
        return

    engine = create_engine()
    try:
        inserted = 0
        skipped = 0

        with engine.begin() as conn:
            for lesson_file in lesson_files:
                lesson_data = json.loads(lesson_file.read_text(encoding="utf-8"))
                result = conn.execute(
                    sqlalchemy.text("""
                        INSERT INTO materials (type, name, data)
                        SELECT :type, :name, CAST(:data AS jsonb)
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM materials
                            WHERE type = :type AND name = :name
                        )
                        RETURNING id;
                        """),
                    {
                        "type": "lesson",
                        "name": lesson_file.name,
                        "data": json.dumps(lesson_data),
                    },
                )

                if result.fetchone() is None:
                    skipped += 1
                else:
                    inserted += 1

        print(f"Imported {inserted} lesson file(s).")
        print(f"Skipped {skipped} existing lesson file(s).")
    finally:
        engine.dispose()
        connector.close()


if __name__ == "__main__":
    main()
