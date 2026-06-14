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
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector

load_dotenv()

app = FastAPI()

# ── Database connection ──
connector = Connector()


def get_connection():
    return connector.connect(
        os.getenv("INSTANCE_CONNECTION_NAME"),
        "pg8000",
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        db=os.getenv("DB_NAME"),
    )


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
        conn = get_connection()
        cursor = conn.cursor()

        # Extract search terms from question
        search_terms = request.question.lower().split()

        # Build keyword array for postgres && operator, && checks whether two arrays share any elemnt
        keyword_conditions = (
            f"keywords && ARRAY[{",".join(["%s"] * len(search_terms))}]"
        )

        # Full text search across Q_plain, A_plain, content_plain
        text_search = " OR ".join([f"""(
                LOWER("Q_plain") LIKE %s OR
                LOWER("A_plain") LIKE %s OR
                LOWER(content_plain) LIKE %s
            )""" for _ in search_terms])

        # Build kind filter
        kind_filter = "AND kind = %s" if request.kind else ""

        query = f"""
            SELECT 
                id, kind, name, keywords,
                "Q_plain", "A_plain", content_plain,
                -- Simple relevance score: keyword matches weighted higher
                (
                    CASE WHEN keywords && ARRAY[{','.join(['%s'] * len(search_terms))}]::varchar[] 
                    THEN 0.6 ELSE 0.0 END +
                    CASE WHEN LOWER("Q_plain") LIKE ANY(ARRAY[{','.join(['%s'] * len(search_terms))}])
                    THEN 0.4 ELSE 0.0 END
                ) as score
            FROM chunks
            WHERE ({keyword_conditions} OR {text_search})
            {kind_filter}
            ORDER BY score DESC
            LIMIT %s
        """

        # Build params
        like_terms = [f"%{t}%" for t in search_terms]
        params = (
            search_terms  # keyword && conditions
            + like_terms * 3  # text search LIKE
            + search_terms  # score keyword check
            + like_terms  # score Q_plain check
            + ([request.kind] if request.kind else [])
            + [request.top_k]
        )

        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

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
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT kind, COUNT(*) as count FROM chunks GROUP BY kind ORDER BY count DESC"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"kinds": [{"kind": r["kind"], "count": r["count"]} for r in rows]}


# ── Serve frontend ──
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}

