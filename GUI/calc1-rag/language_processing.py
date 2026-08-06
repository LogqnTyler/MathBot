from __future__ import annotations

import os
import threading
from typing import Any, Mapping, Sequence

from sentence_transformers import SentenceTransformer

# ── Embedding model (used for RAG retrieval) ──
model = SentenceTransformer("BAAI/bge-m3")


def embed_query(text: str | list[str]) -> list[float]:
    if isinstance(text, str):
        text = text
    return model.encode_query(text).tolist()


# ── Generation model (Qwen2.5-Math, used to actually generate responses) ──
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-Math-1.5B-Instruct")

# 128 tokens is enough for a short answer, but not a full problem + step-by-step
# solution. Bump this up; override via env var if generations run too slow.
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "128"))





# ── Prompt assembly (RAG context -> a single prompt string for the model) ──
def generate_prompt_internal(
    student_prompt: str,
    *,
    practice_problems: Sequence[Mapping[str, Any]] | None = None,
    extra_contents: Sequence[Mapping[str, Any]] | None = None,
    definitions: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    parts = [
        "You are a helpful chatbot that generates practice problems for an introductory level college calculus class. You make practice problems with step by step solutions. You will create a practice problem based on some example practice problems, other course content, and, if applicable, definitions."
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
            'Now generate a practice problem as well as a step-by-step solution to the problem according to the following prompt from the student. Once you generate the problem, write the solution in a step by step format, correcting any mistakes you make along the way. When the correct solution is found, reply with "# Problem: {problem goes here} # Solution: {solution goes here}" and then immediately terminate your response.',
            f"Student: {student_prompt}",
            "Chatbot:",
        ]
    )

    return "\n\n".join(parts)
