from __future__ import annotations

from typing import Any, Mapping, Sequence

from sentence_transformers import SentenceTransformer

# ── Embedding model (used for RAG retrieval) ──
model = SentenceTransformer("BAAI/bge-m3", device="cpu")


def embed_query(text: str | list[str]) -> list[float]:
    if isinstance(text, str):
        text = text
    return model.encode_query(text).tolist()


# ── Prompt assembly (RAG context -> a single prompt string for the model) ──
def generate_prompt_internal(
    student_prompt: str,
    *,
    include_solution: bool = True,
    practice_problems: Sequence[Mapping[str, Any]] | None = None,
    extra_contents: Sequence[Mapping[str, Any]] | None = None,
    definitions: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    parts = [
        "You are a helpful chatbot that generates practice problems for an "
        "introductory college calculus class. Follow the requested solution "
        "setting exactly. Use the supplied examples, course content, and "
        "definitions when relevant."
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
        if answer and include_solution:
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

    if include_solution:
        task_instruction = (
            "Generate a practice problem and a step-by-step solution. Include a "
            "clear final answer under the exact heading **Final Answer:**. Never "
            "write [[FINAL ANSWER]] or put brackets around that heading. Then "
            "terminate the response."
        )
    else:
        task_instruction = (
            "Generate only the practice problem statement. Do not reveal a "
            "solution, final answer, hint, setup, or answer key. End immediately "
            "after the problem statement."
        )

    parts.extend([task_instruction, f"Student: {student_prompt}", "Chatbot:"])

    return "\n\n".join(parts)


def generate_follow_up_prompt(
    student_prompt: str,
    *,
    conversation: Sequence[Mapping[str, str]],
    related_chunks: Sequence[Mapping[str, Any]] | None = None,
    definitions: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    parts = [
        "You are continuing a conversation with a calculus student. Answer the "
        "student's follow-up question directly. Use the existing practice problem "
        "and solution from the conversation. Do not replace them with a new problem "
        "unless the student explicitly asks for one."
    ]

    if related_chunks:
        parts.append("# Related course material")
        for chunk in related_chunks:
            content = (
                chunk.get("content_plain")
                or chunk.get("problem_context_plain")
                or chunk.get("Q_plain")
            )
            if content:
                parts.append(str(content))

    for definition in definitions or []:
        content = definition.get("content_plain")
        if content:
            parts.append(str(content))

    parts.append("# Conversation")
    for message in conversation:
        role = "Student" if message["role"] == "user" else "MathBot"
        parts.append(f"{role}: {message['content']}")

    parts.extend(
        [
            f"Student follow-up: {student_prompt}",
            "Answer the follow-up clearly and concisely. Use LaTeX delimiters "
            "around every mathematical expression.",
            "MathBot:",
        ]
    )
    return "\n\n".join(parts)
