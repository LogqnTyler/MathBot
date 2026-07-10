from contextlib import asynccontextmanager
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from embedding import embed_query
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
class SimilarityQuery(BaseModel):
    question: str
    similarity_threshold: float = 0.25
    kind: Optional[str] = "problem"
    top_k: int = 5


class KeywordQuery(BaseModel):
    keyword: str
    kind: Optional[str] = "definition"


@app.post("/retrieve")
async def retrieve_similarity(request: SimilarityQuery) -> list[dict[str, Any]]:
    embedding = embed_query(request.question)
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

    softmax_temp = 2.0

    scores = np.array([chunk["score"] for chunk in chunks])
    # Adding some randomness to how we choose our retrival chunks, so we don't
    # end up with the same chunks every time for a request.
    chosen_chunks = np.random.choice(
        chunks,
        size=min(request.top_k, len(chunks)),
        replace=False,
        p=softmax(
            scores * softmax_temp
        ),  # adding temp to weight more similar chunks higher
    )
    return chosen_chunks.tolist()


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
