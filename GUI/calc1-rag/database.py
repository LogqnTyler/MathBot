import os
from contextlib import contextmanager
from typing import Any

import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# load secrets
from dotenv import load_dotenv

load_dotenv()


engine: sa.Engine | None = None
SessionLocal: sessionmaker | None = None
Base = declarative_base()

CHUNKS_COLUMNS = """
    id,
    kind,
    mat_id,
    name,
    learning_objective,
    content_plain,
    content_latex,
    problem_context_plain,
    problem_context_latex,
    sub_prob_part,
    "Q_plain",
    "Q_latex",
    "A_plain",
    "A_latex",
    keywords
"""


def _init_db_hosted(database_url: str) -> None:
    """
    Connect to a hosted Postgres database such as Supabase.
    """
    global engine, SessionLocal

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    SessionLocal = sessionmaker(bind=engine)


def _init_db_local() -> None:
    """
    Connect directly to the local Docker Postgres instance.
    """
    global engine, SessionLocal

    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    db_user = os.environ.get("DB_USER", "postgres")
    db_pass = os.environ.get("DB_PASSWORD") or os.environ.get(
        "DB_PASS",
        "postgres",
    )
    db_name = os.environ.get("DB_NAME", "postgres")

    url = (
        f"postgresql+pg8000://"
        f"{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    )

    engine = create_engine(
        url,
        pool_pre_ping=True,
    )
    SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """
    Use DATABASE_URL for hosted Postgres/Supabase.
    Otherwise fall back to local Docker Postgres.
    """
    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        print("DATABASE_URL set — connecting to hosted Postgres.")
        _init_db_hosted(database_url)
    else:
        print("DATABASE_URL not set — connecting to local Postgres instead.")
        _init_db_local()


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
            {CHUNKS_COLUMNS},
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

    return [dict(row) for row in rows]


def select_all_keywords() -> list[str]:
    """Return distinct keywords across all chunks."""
    stmt = sa.text("""
        SELECT DISTINCT chunk_keyword.value AS keyword
        FROM chunks
        CROSS JOIN LATERAL unnest(keywords) AS chunk_keyword(value)
        WHERE chunk_keyword.value IS NOT NULL
        ORDER BY chunk_keyword.value
        """)

    with get_engine().connect() as conn:
        rows = conn.execute(stmt).scalars().all()

    return list(rows)


def select_chunks_by_keyword(keyword: str, kind: str | None = None) -> list[dict[str, Any]]:
    kind_filter = "AND kind = :kind" if kind is not None else ""
    stmt = sa.text(f"""
        SELECT
            {CHUNKS_COLUMNS}
        FROM chunks
        WHERE keywords IS NOT NULL
          {kind_filter}
          AND EXISTS (
              SELECT 1
              FROM unnest(keywords) AS chunk_keyword(value)
              WHERE lower(chunk_keyword.value) = lower(:keyword)
          )
        ORDER BY id
        """)

    params = {"keyword": keyword}
    if kind is not None:
        params["kind"] = kind

    with get_engine().connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()

    return [dict(row) for row in rows]


def select_chunks_by_keywords(keywords: list[str], kind: str | None = None) -> list[dict[str, Any]]:
    """
    Like select_chunks_by_keyword, but matches any chunk whose keywords array
    overlaps with the given list (case-insensitive). Used when a topic maps
    to several underlying keywords rather than a single exact one.
    """
    if not keywords:
        return []

    kind_filter = "AND kind = :kind" if kind is not None else ""
    stmt = sa.text(f"""
        SELECT
            {CHUNKS_COLUMNS}
        FROM chunks
        WHERE keywords IS NOT NULL
          {kind_filter}
          AND EXISTS (
              SELECT 1
              FROM unnest(keywords) AS chunk_keyword(value)
              WHERE lower(chunk_keyword.value) = ANY(:keyword_list)
          )
        ORDER BY id
        """)

    params: dict[str, Any] = {"keyword_list": [kw.lower() for kw in keywords]}
    if kind is not None:
        params["kind"] = kind

    with get_engine().connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()

    return [dict(row) for row in rows]


def close_db() -> None:
    global engine, SessionLocal

    if engine is not None:
        engine.dispose()
        engine = None
        SessionLocal = None


@contextmanager
def db_lifespan():
    init_db()
    try:
        yield
    finally:
        close_db()
