from __future__ import annotations

import os
import threading
from typing import Any, Mapping, Sequence

# ── Embedding model (Gemini Embedding via Vertex AI) ──
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSION = 1024


def embed_query(text: str) -> list[float]:
    """Create a 1024-dimensional retrieval-query embedding using Vertex AI."""

    import json
    import urllib.request

    import google.auth
    from google.auth.transport.requests import Request

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "as-math-rag-tool-9d28")
    location = os.getenv("VERTEX_LOCATION", "us-east4")

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/publishers/google/"
        f"models/{EMBEDDING_MODEL}:predict"
    )

    payload = {
        "instances": [
            {
                "content": text,
                "task_type": "RETRIEVAL_QUERY",
            }
        ],
        "parameters": {
            "outputDimensionality": EMBEDDING_DIMENSION,
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))

    try:
        return result["predictions"][0]["embeddings"]["values"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Vertex AI returned an invalid embedding response: {result}"
        ) from exc


# ── Generation model (Gemini 2.5 Flash-Lite via Vertex AI) ──
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "as-math-rag-tool-9d28")
GCP_LOCATION = os.getenv("VERTEX_LOCATION", "us-east4")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))


def generate_gemini_response(
    prompt: str,
    *,
    response_schema: dict[str, Any] | None = None,
) -> str:
    """Generate a MathBot response using Gemini on Vertex AI.

    When response_schema is supplied, require Gemini to return JSON
    conforming to that schema.
    """

    import json
    import urllib.request

    import google.auth
    from google.auth.transport.requests import Request

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())

    url = (
        f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{GCP_PROJECT}/locations/{GCP_LOCATION}/publishers/google/"
        f"models/{GEMINI_MODEL}:generateContent"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
        },
    }

    if response_schema is not None:
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseSchema"] = response_schema

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {result}")

    parts = candidates[0].get("content", {}).get("parts", [])
    output = "".join(part.get("text", "") for part in parts).strip()

    if not output:
        raise RuntimeError(f"Gemini returned an empty response: {result}")

    return output





# ── Prompt assembly (RAG context -> a single prompt string for the model) ──
def generate_prompt_internal(
    student_prompt: str,
    *,
    practice_problems: Sequence[Mapping[str, Any]] | None = None,
    extra_contents: Sequence[Mapping[str, Any]] | None = None,
    definitions: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    parts = [
        "You are MathBot, a tutor for an introductory college single-variable calculus course. "
        "Use the course material below only as mathematical and curricular context. "
        "Follow the student's requested topic exactly. "
        "IMPORTANT COURSE SCOPE: This course does not cover trigonometric functions. "
        "Do not introduce sine, cosine, tangent, inverse trigonometric functions, "
        "trigonometric identities, or trig-based examples, exercises, quiz questions, "
        "applications, derivatives, or integrals. Use non-trigonometric functions instead. "
        "Do not assume that the task is a practice problem or that a solution should be provided; "
        "the final task instructions determine the required response."
    ]

    for index, problem in enumerate(practice_problems or [], start=1):
        parts.append(f"# Practice Problem {index}")

        context = problem.get("problem_context_plain") or problem.get("context")
        if context:
            parts.append(f"Context: {context}")

        question = problem.get("Q_plain") or problem.get("question") or problem.get("content_plain")
        answer = problem.get("A_plain") or problem.get("answer")

        if question:
            parts.append(f"Question: {question}")
        if answer:
            parts.append(f"Answer: {answer}")

    if extra_contents:
        parts.append("# extra related content")
        for content in extra_contents:
            content_type = content.get("kind") or content.get("type")
            material = content.get("content_plain") or content.get("material")

            if content_type:
                parts.append(f"Type: {content_type}")
            if material:
                parts.append(f"Material:\n\n{material}")

    for definition in definitions or []:
        term = definition.get("name") or definition.get("defined_term") or definition.get("term")
        content = definition.get("content_plain") or definition.get("definition")

        if term or content:
            parts.append("# Definition")
        if term:
            parts.append(f"DefinedTerm: {term}")
        if content:
            parts.append(f"Definition: {content}")

    parts.extend(
        [
            "# Student Request",
            student_prompt,
            (
                "Use the course material above as supporting context. "
                "Follow the mandatory final task instructions that appear after this context."
            ),
        ]
    )

    return "\n\n".join(parts)
