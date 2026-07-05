from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from embedding import embed_text
import numpy as np
from scipy.special import softmax

from db_orm import db_lifespan, get_engine, query_similar_chunks
import sqlalchemy as sa


@asynccontextmanager
async def lifespan(app: FastAPI):

    # ── Serve frontend ──
    app.mount("/static", StaticFiles(directory="static"), name="static")

    with db_lifespan():
        yield


app = FastAPI(lifespan=lifespan)


# ── Request/Response models ──
class QueryRequest(BaseModel):
    question: str
    kind: Optional[str] = None
    similarity_threshold: float = 0.25
    top_k: int = 5


@app.post("/retrieve")
async def retrieve(request: QueryRequest) -> list[dict[str, Any]]:
    embedding = embed_text(request.question)
    chunks = query_similar_chunks(
        embedding,
        kind=request.kind,
        min_score=request.similarity_threshold,
    )
    ## DEBUG
    print(
        f"Found {len(chunks)} chunks with similarity > {request.similarity_threshold}"
    )
    if not chunks:
        return []

    scores = [chunk["score"] for chunk in chunks]
    # Adding some randomness to how we choose our retrival chunks, so we don't
    # end up with the same chunks every time for a request.
    chosen_chunks = np.random.choice(
        chunks,
        size=min(request.top_k, len(chunks)),
        replace=False,
        p=softmax(scores),
    )
    return chosen_chunks.tolist()


# # ── Search endpoint ──
# async def query_chunks(request: QueryRequest):
#     try:
#         chunks = [
#             {
#                 "id": row["id"],
#                 "kind": row["kind"],
#                 "name": row["name"],
#                 "keywords": row["keywords"] or [],
#                 "Q_plain": row["Q_plain"],
#                 "A_plain": row["A_plain"],
#                 "content_plain": row["content_plain"],
#                 "score": float(row["score"]) if row["score"] else 0.0,
#             }
#             for row in await retrieval(request)
#         ]
#
#         return {"chunks": chunks, "count": len(chunks)}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# ── Filter by kind ──
@app.get("/kinds")
async def get_kinds():
    engine = get_engine()
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
