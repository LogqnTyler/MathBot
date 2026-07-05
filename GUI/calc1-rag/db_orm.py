import os
from contextlib import contextmanager
from typing import Any

import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from google.cloud.sql.connector import Connector, IPTypes

# load secrets
from dotenv import load_dotenv

load_dotenv()


connector: Connector | None = None
engine: sa.Engine | None = None
SessionLocal: sessionmaker | None = None
Base = declarative_base()


def init_db():
    global connector, engine, SessionLocal

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

    # start sql orm
    engine = create_engine("postgresql+pg8000://", creator=getconn)
    SessionLocal = sessionmaker(bind=engine)


def get_engine() -> sa.Engine:
    if engine is None:
        raise RuntimeError("Database has not been initialized yet")
    return engine


def get_sessionmaker() -> sessionmaker:
    if SessionLocal is None:
        raise RuntimeError("Database has not been initialized yet")
    return SessionLocal


def _vector_literal(values: list[float]) -> str:
    if not values:
        raise ValueError("Embedding vector cannot be empty")
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def query_similar_chunks(
    query_embedding: list[float],
    *,
    kind: str | None = None,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Return chunks with cosine similarity to query_embedding greater than min_score."""
    kind_filter = "AND kind = :kind" if kind is not None else ""
    stmt = sa.text(f"""
        SELECT
            id,
            kind,
            name,
            keywords,
            "Q_plain",
            "A_plain",
            content_plain,
            problem_context_plain,
            1 - (embedding <=> CAST(:embedding AS vector)) AS score
        FROM chunks
        WHERE embedding IS NOT NULL
          {kind_filter}
          AND 1 - (embedding <=> CAST(:embedding AS vector)) > :min_score
        ORDER BY embedding <=> CAST(:embedding AS vector)
        """)

    params = {
        "embedding": _vector_literal(query_embedding),
        "min_score": min_score,
    }
    if kind is not None:
        params["kind"] = kind

    with get_engine().connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()

    return [
        {
            "id": row["id"],
            "kind": row["kind"],
            "name": row["name"],
            "keywords": row["keywords"] or [],
            "Q_plain": row["Q_plain"],
            "A_plain": row["A_plain"],
            "content_plain": row["content_plain"],
            "problem_context_plain": row["problem_context_plain"],
            "score": float(row["score"]) if row["score"] is not None else 0.0,
        }
        for row in rows
    ]


def close_db() -> None:
    global connector, engine, SessionLocal

    if engine is not None:
        engine.dispose()
        engine = None
        SessionLocal = None

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
