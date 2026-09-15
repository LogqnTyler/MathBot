from __future__ import annotations

import json
import os
import urllib.request

import google.auth
from google.auth.transport.requests import Request


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSION = 1024


def embed_doc(text: str) -> list[float]:
    """Create a 1024-dimensional document embedding using Vertex AI."""

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
                "task_type": "RETRIEVAL_DOCUMENT",
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
