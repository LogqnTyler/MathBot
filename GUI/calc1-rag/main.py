import os
from contextlib import asynccontextmanager

import string
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from embedding import embed_text

from db_orm import db_lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):

    # ── Serve frontend ──
    app.mount("/static", StaticFiles(directory="static"), name="static")

    with db_lifespan():
        yield



app = FastAPI(lifespan=lifespan)



@app.on_event("shutdown")
def closeout():
    engine.dispose()
    connector.close()


# ── Request/Response models ──
class QueryRequest(BaseModel):
    question: str
    kind: Optional[str] = None
    top_k: int = 5


class Chunk(BaseModel): # i think this should be deleted and replaced with something else
    id: int
    kind: str
    name: str
    keywords: list[str]
    Q_plain: Optional[str]
    A_plain: Optional[str]
    content_plain: Optional[str]
    score: float


async def retrieval(request: QueryRequest) -> list[Chunk]:
    embedding = embed_text(request.question)

    with engine.connect() as conn:

    

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
            rows = conn.execute(sa.text(query), params).mappings().all()

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
                sa.text(
                    "SELECT kind, COUNT(*) as count FROM chunks GROUP BY kind ORDER BY count DESC"
                )
            )
            .mappings()
            .all()
        )
    return {"kinds": [{"kind": r["kind"], "count": r["count"]} for r in rows]}




@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}



