from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

import json
from pathlib import Path
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


# ── Lesson topics (from JSON/lessonN.json) ──
JSON_DIR = Path("JSON")


def load_topics() -> list[dict[str, Any]]:
    """
    Scan JSON_DIR for lessonN.json files and return topic metadata
    (id, name, week) sorted by week. `id` is the filename stem, e.g.
    "lesson1", so the frontend can reference a topic unambiguously
    even if two lessons ever share the same name.
    """
    topics: list[dict[str, Any]] = []
    if not JSON_DIR.exists():
        print(f"WARNING: JSON topic directory '{JSON_DIR}' does not exist.")
        return topics

    for file in sorted(JSON_DIR.glob("lesson*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not read {file}: {e}")
            continue

        topics.append(
            {
                "id": file.stem,  # e.g. "lesson1"
                "name": data.get("name", file.stem),
                "week": data.get("week"),
            }
        )

    topics.sort(key=lambda t: (t["week"] is None, t["week"]))
    return topics


@asynccontextmanager
async def lifespan(app: FastAPI):
    global KEYWORDS, TOPICS

    # ── Serve frontend ──
    app.mount("/static", StaticFiles(directory="static"), name="static")

    TOPICS = load_topics()

    with db_lifespan():
        KEYWORDS = select_all_keywords()
        yield


app = FastAPI(lifespan=lifespan)


CHUNK_TYPES = ("problem", "definition", "other_material")
ChunkKind = Literal[*CHUNK_TYPES]

REQUEST_TYPES = ("practice_problems", "alternate_explanations", "concept_summary", "quiz_me")
RequestKind = Literal[*REQUEST_TYPES]

REQUEST_TYPE_TEMPLATES: dict[str, str] = {
    "practice_problems": "Generate a practice problem about {subject}.",
    "alternate_explanations": "Give an alternate explanation of {subject}, using a different approach, framing, or analogy than a typical textbook.",
    "concept_summary": "Provide a concise summary of the key concepts and definitions for {subject}.",
    "quiz_me": "Quiz me with a short question about {subject} to check my understanding, then wait for my answer.",
}

KEYWORDS: list[str] = []
TOPICS: list[dict[str, Any]] = []
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
    request_type: RequestKind = "practice_problems"
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


@app.get("/topics")
async def get_topics() -> list[dict[str, Any]]:
    """Returns lesson topics (id, name, week) for the topic picker, sorted by week."""
    return TOPICS


@app.post("/generate_prompt")
def generate_prompt(request: GeneratePrompt) -> dict[str, Any]:
    if request.subject not in KEYWORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid subject '{request.subject}'. Must be one of: {KEYWORDS}",
        )

    template = REQUEST_TYPE_TEMPLATES[request.request_type]
    student_prompt = template.format(subject=request.subject)
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
