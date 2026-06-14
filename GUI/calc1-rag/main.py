# import os
# from fastapi import FastAPI, HTTPException
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from typing import Optional
# import psycopg2
# from psycopg2.extras import RealDictCursor
# from dotenv import load_dotenv

# load_dotenv()

# app = FastAPI()

# # ── Database connection ──
# def get_connection():
#     return psycopg2.connect(
#         host=os.getenv("DB_HOST"),
#         database=os.getenv("DB_NAME"),
#         user=os.getenv("DB_USER"),
#         password=os.getenv("DB_PASSWORD"),
#         cursor_factory=RealDictCursor
#     )

# # ── Request model ──
# class QueryRequest(BaseModel):
#     question: str
#     kind: Optional[str] = None
#     top_k: int = 5

# # ── Strip LaTeX ──
# def strip_latex(text: str) -> str:
#     if not text:
#         return text
#     return (text
#         .replace("$$", "")
#         .replace("$", "")
#         .replace("\\(", "")
#         .replace("\\)", "")
#         .replace("\\[", "")
#         .replace("\\]", "")
#         .strip()
#     )

# # ── Search endpoint ──
# @app.post("/query")
# async def query_chunks(request: QueryRequest):
#     try:
#         conn = get_connection()
#         cursor = conn.cursor()

#         search_terms = [t.lower() for t in request.question.split()]
#         like_terms = [f"%{t}%" for t in search_terms]

#         kind_filter = "AND kind = %(kind)s" if request.kind else ""

#         query = f"""
#             SELECT
#                 id, kind, name, keywords,
#                 "Q_plain", "A_plain", content_plain,
#                 (
#                     CASE WHEN keywords && %(keywords)s::varchar[] THEN 0.6 ELSE 0.0 END +
#                     CASE WHEN LOWER("Q_plain") LIKE ANY(%(like_terms)s) THEN 0.3 ELSE 0.0 END +
#                     CASE WHEN LOWER("A_plain") LIKE ANY(%(like_terms)s) THEN 0.1 ELSE 0.0 END
#                 ) as score
#             FROM chunks
#             WHERE (
#                 keywords && %(keywords)s::varchar[]
#                 OR LOWER("Q_plain") LIKE ANY(%(like_terms)s)
#                 OR LOWER("A_plain") LIKE ANY(%(like_terms)s)
#                 OR LOWER(content_plain) LIKE ANY(%(like_terms)s)
#             )
#             {kind_filter}
#             ORDER BY score DESC
#             LIMIT %(top_k)s
#         """

#         cursor.execute(query, {
#             "keywords": search_terms,
#             "like_terms": like_terms,
#             "kind": request.kind,
#             "top_k": request.top_k
#         })

#         rows = cursor.fetchall()
#         cursor.close()
#         conn.close()

#         chunks = [
#             {
#                 "id": str(row["id"]),
#                 "kind": row["kind"],
#                 "name": row["name"],
#                 "keywords": row["keywords"] or [],
#                 "Q_plain": strip_latex(row["Q_plain"]),
#                 "A_plain": strip_latex(row["A_plain"]),
#                 "content_plain": strip_latex(row["content_plain"]),
#                 "score": float(row["score"]) if row["score"] else 0.1,
#                 "topic": row["kind"].replace("-", " ").title()
#             }
#             for row in rows
#         ]

#         return {"chunks": chunks, "count": len(chunks)}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ── Serve frontend ──
# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/")
# async def root():
#     return FileResponse("static/index.html")

# @app.get("/health")
# async def health():
#     return {"status": "ok"}

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy
from sqlalchemy import create_engine
import string

load_dotenv()

app = FastAPI()

# ── Database connection ──
connector = Connector(refresh_strategy="LAZY")


def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """Initializes a SQLAlchemy connection pool for Cloud SQL Postgres."""
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

    return create_engine("postgresql+pg8000://", creator=getconn)


engine = connect_with_connector()


@app.on_event("shutdown")
def shutdown_db_connections():
    engine.dispose()
    connector.close()


# ── Request/Response models ──
class QueryRequest(BaseModel):
    question: str
    kind: Optional[str] = None
    top_k: int = 5


class Chunk(BaseModel):
    id: int
    kind: str
    name: str
    keywords: list[str]
    Q_plain: Optional[str]
    A_plain: Optional[str]
    content_plain: Optional[str]
    score: float


# ── Search endpoint ──
@app.post("/query")
async def query_chunks(request: QueryRequest):
    try:
        # Extract search terms from question, remove punctuation
        cleaned_question = request.question.translate(
            str.maketrans("", "", string.punctuation)
        )
        # DEBUG: printing the cleaned question to make sure it has no punctuation
        print(f"Cleaned question: '{cleaned_question}'")
        search_terms = cleaned_question.lower().split()

        # Build kind filter
        kind_filter = "AND kind = :kind" if request.kind else ""

        query = f"""
            SELECT 
                id, kind, name, keywords,
                "Q_plain", "A_plain", content_plain, 
                -- Simple relevance score: keyword matches weighted higher
                (
                    CASE WHEN keywords && CAST(:keywords AS varchar[]) 
                    THEN 0.6 ELSE 0.0 END +
                    CASE WHEN LOWER("Q_plain") LIKE ANY(CAST(:like_terms AS varchar[]))
                    THEN 0.4 ELSE 0.0 END
                ) as score
            FROM chunks
            WHERE (
                keywords && CAST(:keywords AS varchar[])
                OR LOWER("Q_plain") LIKE ANY(CAST(:like_terms AS varchar[]))
                OR LOWER("A_plain") LIKE ANY(CAST(:like_terms AS varchar[]))
                OR LOWER(content_plain) LIKE ANY(CAST(:like_terms AS varchar[]))
                OR LOWER(problem_context_plain) LIKE ANY(CAST(:like_terms AS varchar[]))
            )
            {kind_filter}
            ORDER BY score DESC
            LIMIT :top_k
        """

        # Build params
        like_terms = [f"%{t}%" for t in search_terms]
        params = {
            "keywords": search_terms,
            "like_terms": like_terms,
            "kind": request.kind,
            "top_k": request.top_k,
        }

        with engine.connect() as conn:
            rows = conn.execute(sqlalchemy.text(query), params).mappings().all()

        chunks = [
            {
                "id": str(row["id"]),
                "kind": row["kind"],
                "name": row["name"],
                "keywords": row["keywords"] or [],
                "Q_plain": row["Q_plain"],
                "A_plain": row["A_plain"],
                "content_plain": row["content_plain"],
                "score": float(row["score"]) if row["score"] else 0.0,
                "topic": row["kind"].replace("-", " ").title(),
            }
            for row in rows
        ]

        return {"chunks": chunks, "count": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Filter by kind ──
@app.get("/kinds")
async def get_kinds():
    with engine.connect() as conn:
        rows = (
            conn.execute(
                sqlalchemy.text(
                    "SELECT kind, COUNT(*) as count FROM chunks GROUP BY kind ORDER BY count DESC"
                )
            )
            .mappings()
            .all()
        )
    return {"kinds": [{"kind": r["kind"], "count": r["count"]} for r in rows]}


# ── Serve frontend ──
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
