import os
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy import create_engine
from google.cloud.sql.connector import Connector, IPTypes

# load secrets
from dotenv import load_dotenv

load_dotenv()


connector: Connector | None = None
engine: sa.Engine | None = None


def init_db():
    global connector, engine

    connector = Connector(refresh_strategy="LAZY")

    """Initializes a sa connection pool for Cloud SQL Postgres."""
    instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]
    db_user = os.environ["DB_USER"]
    db_pass = os.environ.get("DB_PASSWORD") or os.environ["DB_PASS"]
    db_name = os.environ["DB_NAME"]
    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    def getconn():
        return connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type,
        )

    engine = create_engine("postgresql+pg8000://", creator=getconn)


def get_engine() -> sa.Engine:
    if engine is None:
        RuntimeError("Database has noe been initialized yet")
    return engine


def close_db() -> None:
    global connector, engine

    if engine is not None:
        engine.dispose()
        engine = None

    if connector is not None:
        connector.close()
        connector = None


@contextmanager
def db_lifespan():
    init_db()
    try:
        yield
    finally:
        close_db()
