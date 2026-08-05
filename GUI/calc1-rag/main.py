from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from typing import Any, Literal
from pydantic import BaseModel, Field

import numpy as np
from scipy.special import softmax

from dotenv import load_dotenv

load_dotenv()

from language_processing import embed_query, generate_prompt_internal
from database import (
    db_lifespan,
    query_similar_chunks,
    select_all_keywords,
    select_chunks_by_keyword,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global KEYWORDS

    # ── Serve frontend ──
    app.mount("/static", StaticFiles(directory="static"), name="static")

    with db_lifespan():
        KEYWORDS = select_all_keywords()
        yield


app = FastAPI(lifespan=lifespan)


CHUNK_TYPES = ("problem", "definition", "other_material")
ChunkKind = Literal[*CHUNK_TYPES]
KEYWORDS: list[str] = []
PROMPT_SIMILARITY_THRESHOLD = 0.25
PROMPT_PROBLEM_COUNT = 3
PROMPT_EXTRA_CONTENT_COUNT = 3


# ── Request/Response models ──
class SimilarityQuery(BaseModel):
    question: str
    similarity_threshold: float = 0.25
    kind: ChunkKind = "problem"
    top_k: int = 5


class KeywordQuery(BaseModel):
    keyword: str
    kind: ChunkKind = "definition"


class GeneratePrompt(BaseModel):
    subject: str
    other_details: str = Field(
        default="",
        min_length=0,
        max_length=120,
        description="Additional details for the problem",
    )


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


@app.post("/generate_prompt")
def generate_prompt(request: GeneratePrompt) -> dict[str, Any]:
    if request.subject not in KEYWORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid subject '{request.subject}'. Must be one of: {KEYWORDS}",
        )

    student_prompt = f"Generate a practice problem about {request.subject}."
    if request.other_details:
        student_prompt = f"{student_prompt} {request.other_details}"

    embedding = embed_query(student_prompt)
    practice_problems = query_similar_chunks(
        embedding,
        kind="problem",
        min_score=PROMPT_SIMILARITY_THRESHOLD,
    )[:PROMPT_PROBLEM_COUNT]
    extra_contents = query_similar_chunks(
        embedding,
        kind="other_material",
        min_score=PROMPT_SIMILARITY_THRESHOLD,
    )[:PROMPT_EXTRA_CONTENT_COUNT]
    definitions = select_chunks_by_keyword(request.subject, kind="definition")

    return {
        "prompt": generate_prompt_internal(
            student_prompt=student_prompt,
            practice_problems=practice_problems,
            extra_contents=extra_contents,
            definitions=definitions,
        )
    }


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
