from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field
from scipy.special import softmax

load_dotenv()

from database import (
    db_lifespan,
    query_similar_chunks,
    select_all_keywords,
    select_chunks_by_keywords,
)
from language_processing import (
    embed_query,
    generate_follow_up_prompt,
    generate_prompt_internal,
)
from qwen_model import (
    generate_qwen_response,
    load_qwen_model,
)


# ── Lesson topics (from JSON/lessonN.json) ──

JSON_DIR = Path("JSON")


def load_topics() -> list[dict[str, Any]]:
    """
    Scan JSON_DIR for lessonN.json files and return topic metadata
    (id, name, week) sorted by week.

    The id is the filename stem, such as "lesson1".
    """
    topics: list[dict[str, Any]] = []

    if not JSON_DIR.exists():
        print(f"WARNING: JSON topic directory '{JSON_DIR}' does not exist.")
        return topics

    for file in sorted(JSON_DIR.glob("lesson*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: could not read {file}: {exc}")
            continue

        topics.append(
            {
                "id": file.stem,
                "name": data.get("name", file.stem),
                "week": data.get("week"),
            }
        )

    topics.sort(key=lambda topic: (topic["week"] is None, topic["week"]))
    return topics


@asynccontextmanager
async def lifespan(app: FastAPI):
    global KEYWORDS, TOPICS

    app.mount(
        "/static",
        StaticFiles(directory="static"),
        name="static",
    )

    TOPICS = load_topics()

    # Load Qwen once when FastAPI starts.
    load_qwen_model()

    with db_lifespan():
        KEYWORDS = select_all_keywords()
        yield


app = FastAPI(lifespan=lifespan)


# ── Request types ──

CHUNK_TYPES = (
    "problem",
    "definition",
    "other_material",
)
ChunkKind = Literal[*CHUNK_TYPES]

REQUEST_TYPES = (
    "practice_problems",
    "alternate_explanations",
    "concept_summary",
    "quiz_me",
)
RequestKind = Literal[*REQUEST_TYPES]


REQUEST_TYPE_TEMPLATES: dict[str, str] = {
    "practice_problems": (
        "Generate one complete practice problem about {subject}. "
        "Follow any requested real-world context exactly."
    ),
    "alternate_explanations": (
        "Give an alternate explanation of {subject}. "
        "Use a different approach, framing, application, or analogy "
        "than a typical textbook. Follow any requested context exactly."
    ),
    "concept_summary": (
        "Provide a concise summary of the key concepts and definitions "
        "for {subject}. Follow any requested context or application."
    ),
    "quiz_me": (
        "Quiz me with one short question about {subject} to check my "
        "understanding. Follow any requested context exactly. "
        "Ask the question and then wait for my answer."
    ),
}


KEYWORDS: list[str] = []
TOPICS: list[dict[str, Any]] = []


# Maps each lesson topic to relevant database keywords.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Slopes and Rates of Change": [
        "slope",
        "average rate of change",
        "rate of change",
        "net change",
        "linear function",
    ],
    "Introduction to Limits": [
        "limit",
        "limits",
        "limit of a function",
        "one-sided limits",
        "dne",
        "table",
        "table of values",
        "function values",
    ],
    "Limits and Continuity": [
        "limit",
        "limits",
        "continuity",
        "continuity theorem",
        "dne",
        "one-sided limits",
    ],
    "Limits at Infinity": [
        "limits at infinity",
        "infinite limits",
        "end behavior",
        "dominant terms",
        "forms",
        "form infinity/c",
        "form infinity/infinity",
        "direct substitution",
        "factoring",
    ],
    "Derivatives & Rates of Change": [
        "derivative",
        "derivative interpretation",
        "instantaneous rate of change",
        "instantaneous velocity",
        "tangent line",
        "secant line",
        "average rate of change",
        "differentiable",
        "units of derivative",
        "difference quotient",
        "leibniz notation",
    ],
    "Basic Differentiation Rules": [
        "power rule",
        "constant rule",
        "constant multiple rule",
        "sum rule",
        "algebraic simplification",
        "derivative formulas",
        "polynomials",
    ],
    "Product & Quotient Rules": [
        "product rule",
        "quotient rule",
        "differentiation rules",
        "evaluating derivatives",
    ],
    "Chain Rule": [
        "chain rule",
        "composite functions",
        "inner function",
        "outer function",
        "function composition",
    ],
    "Derivatives of Exponential & Logarithmic Functions": [
        "derivative of e^x",
        "derivative of a^x",
        "e^x",
        "a^x",
        "logarithmic derivative",
        "exponential derivative",
    ],
    "Properties of Exponential & Logarithmic Functions": [
        "exponent rules",
        "exponent notation",
        "logarithm",
        "logarithms",
        "logarithm properties",
        "logarithmic functions",
        "logarithmic equations",
        "exponential equations",
        "exponential function",
        "exponential functions",
        "exponential expressions",
        "condensing logarithms",
        "natural logarithm",
        "domain",
        "positive exponents",
        "negative exponents",
        "fractional exponents",
        "radical notation",
        "radicals",
        "square roots",
        "inverse functions",
    ],
    "Local Extrema & First Derivative Test": [
        "local maximum",
        "local minimum",
        "first derivative test",
        "critical number",
        "critical numbers",
        "increasing intervals",
        "decreasing intervals",
    ],
    "Properties of Definite Integrals": [
        "additivity",
        "linearity property",
        "reverse limits property",
        "swapping limits",
        "signed area",
    ],
    "Antiderivatives and Indefinite Integrals": [
        "antiderivative",
        "indefinite integral",
        "power rule for antiderivatives",
        "rewriting before integrating",
        "radicals",
        "rational functions",
        "square root functions",
        "power functions",
    ],
    "Concavity & The Second Derivative": [
        "second derivative",
        "concavity",
        "concave up",
        "concave down",
        "critical number",
        "critical numbers",
        "increasing intervals",
        "decreasing intervals",
        "inflection point",
        "sign line",
    ],
    "Summations": [
        "sigma notation",
        "summation notation",
        "summation properties",
        "linearity of summation",
        "writing sums",
        "expanding sums",
        "decomposing sums",
        "evaluating sums",
        "index dependence",
        "break apart property",
        "patterns",
    ],
    "Absolute Extreme Values": [
        "absolute maximum",
        "absolute minimum",
        "closed interval method",
        "closed interval",
        "non-closed interval",
        "open interval",
        "extreme value theorem",
        "first derivative test for absolute extrema",
        "intervals",
        "endpoints",
        "zero-width interval",
    ],
    "Applied Optimization": [
        "optimization",
        "applied optimization",
        "fencing problem",
        "norman window",
        "closed box",
        "area constraint",
        "area maximization",
        "perimeter constraint",
        "volume constraint",
        "cost minimization",
        "marginal cost",
        "total cost",
        "profit",
        "profit function",
        "revenue",
        "revenue function",
        "unit selling price",
        "triangle area",
        "surface area",
        "area of a line segment",
        "domain restrictions",
        "population",
        "predator-prey model",
        "temperature",
        "half-life",
        "linearization",
        "linear approximation",
        "unbounded growth",
    ],
    "Applied Optimization Extra Practice": [
        "optimization",
        "applied optimization",
        "fencing problem",
        "norman window",
        "closed box",
        "area constraint",
        "area maximization",
        "perimeter constraint",
        "volume constraint",
        "cost minimization",
        "marginal cost",
        "total cost",
        "profit",
        "profit function",
        "revenue",
        "revenue function",
        "unit selling price",
        "triangle area",
        "surface area",
        "area of a line segment",
    ],
    "Net Change, Area, and Definite Integrals": [
        "definite integral",
        "integral expression",
        "net area",
        "signed area",
        "units",
        "net change",
    ],
}


# Keep retrieval small for local inference on an 8 GB Mac.
PROMPT_SIMILARITY_THRESHOLD = 0.25
PROMPT_PROBLEM_COUNT = 1
PROMPT_EXTRA_CONTENT_COUNT = 1
PROMPT_DEFINITION_COUNT = 2


# ── Request models ──

class SimilarityQuery(BaseModel):
    question: str
    similarity_threshold: float = 0.25
    kind: ChunkKind = "problem"
    top_k: int = 5


class KeywordQuery(BaseModel):
    keyword: str
    kind: ChunkKind = "definition"


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class GeneratePrompt(BaseModel):
    subject: str
    request_type: RequestKind = "practice_problems"
    include_solution: bool = True
    follow_up: bool = False
    conversation: list[ConversationMessage] = Field(default_factory=list, max_length=8)
    other_details: str = Field(
        default="",
        min_length=0,
        max_length=500,
        description=(
            "The student's original question and any additional requirements "
            "that must be incorporated into the response."
        ),
    )


# ── Retrieval endpoints ──

@app.post("/retrieve")
async def retrieve_similarity(
    request: SimilarityQuery,
) -> list[dict[str, Any]]:
    embedding = embed_query(request.question)

    chunks = query_similar_chunks(
        embedding,
        kind=request.kind,
        min_score=request.similarity_threshold,
    )

    print(
        f"Found {len(chunks)} chunks with similarity "
        f"> {request.similarity_threshold}"
    )

    if not chunks:
        return []

    softmax_temp = 2.0
    scores = np.array([chunk["score"] for chunk in chunks])

    chosen_chunks = np.random.choice(
        chunks,
        size=min(request.top_k, len(chunks)),
        replace=False,
        p=softmax(scores * softmax_temp),
    )

    return chosen_chunks.tolist()


@app.get("/topics")
async def get_topics() -> list[dict[str, Any]]:
    """
    Return lesson topics for the frontend topic picker.
    """
    return TOPICS


# ── Prompt construction ──

def _format_model_output(text: str) -> str:
    """
    Lightly reformat the model output for the frontend.
    """
    text = text.strip()

    text = re.sub(
        r"(?im)^\s*(?:\[\[\s*FINAL ANSWER\s*\]\]:?|"
        r"#{1,6}\s*FINAL ANSWER:?|\*\*FINAL ANSWER:?\*\*|FINAL ANSWER:)\s*",
        "**Final Answer:**\n",
        text,
    )

    text = re.sub(
        r"#\s*Problem:\s*",
        "**Problem:**\n\n",
        text,
        count=1,
    )

    text = re.sub(
        r"#\s*Solution:\s*",
        "\n\n**Solution:**\n\n",
        text,
        count=1,
    )

    return text.strip()


def _source_names(*chunk_groups: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    for chunk in (chunk for group in chunk_groups for chunk in group):
        material_name = chunk.get("material_name")
        if not material_name:
            continue

        lesson_id = Path(material_name).stem
        topic = next((topic for topic in TOPICS if topic["id"] == lesson_id), None)
        if topic is None:
            label = material_name
        else:
            lesson_label = lesson_id.removeprefix("lesson")
            if lesson_label.endswith("extrapractice"):
                lesson_label = lesson_label.removesuffix("extrapractice") + " Extra Practice"
            label = f"Lesson {lesson_label}: {topic['name']}"

        if label not in sources:
            sources.append(label)

    return sources


def _split_problem_solution(text: str) -> tuple[str, str]:
    solution_heading = re.search(
        r"(?im)^\s*(?:\[\[SOLUTION\]\]|#{1,6}\s*(?:worked\s+)?solution:?|"
        r"\*\*(?:worked\s+)?solution:?\*\*)\s*$",
        text,
    )
    if solution_heading is None:
        return _format_model_output(text), ""

    problem = text[: solution_heading.start()]
    solution = text[solution_heading.end() :]
    problem = re.sub(
        r"(?im)^\s*(?:\[\[PROBLEM\]\]|#{1,6}\s*(?:practice\s+)?problem:?|"
        r"\*\*(?:practice\s+)?problem(?:\s+statement)?:?\*\*)\s*$",
        "",
        problem,
        count=1,
    )
    problem = re.sub(r"\s*(?:---\s*)*$", "", problem)
    solution = re.sub(r"^\s*(?:---\s*)*", "", solution)
    return _format_model_output(problem), _format_model_output(solution)


def _requested_context(request: GeneratePrompt) -> str | None:
    """
    Detect an explicitly requested real-world context.

    This prevents generic retrieved examples from overpowering a request
    such as "make this a chemistry story problem."
    """
    details = request.other_details.lower()

    context_aliases: dict[str, tuple[str, ...]] = {
        "chemistry": (
            "chemistry",
            "chemical",
            "reaction",
            "concentration",
            "molar",
            "molecule",
            "compound",
        ),
        "biology": (
            "biology",
            "biological",
            "population",
            "enzyme",
            "cell",
        ),
        "physics": (
            "physics",
            "physical",
            "velocity",
            "acceleration",
            "force",
        ),
        "economics": (
            "economics",
            "economic",
            "profit",
            "revenue",
            "cost",
        ),
        "engineering": (
            "engineering",
            "engineer",
            "design",
            "manufacturing",
        ),
    }

    for context, aliases in context_aliases.items():
        if any(alias in details for alias in aliases):
            return context

    return None


def _build_student_prompt(request: GeneratePrompt) -> str:
    """
    Build a concise primary instruction for retrieval and generation.
    """
    base_request = REQUEST_TYPE_TEMPLATES[request.request_type].format(
        subject=request.subject
    )

    if request.request_type == "practice_problems":
        if request.include_solution:
            base_request += " Include a concise worked solution and final answer."
        else:
            base_request += (
                " Return only the problem statement. Do not include a solution, "
                "answer, hint, or setup."
            )

    details = request.other_details.strip()

    if not details:
        return base_request

    return (
        f"{base_request}\n\n"
        f"Student request: {details}\n\n"
        "The response must directly satisfy the student's request."
    )


def _build_context_directive(
    request: GeneratePrompt,
    student_prompt: str,
    requested_context: str | None,
) -> str:
    """
    Build a short final directive that is appended after the RAG material.

    qwen_model.py uses left-side truncation, so the final directive is the
    portion most likely to survive when the prompt exceeds the token limit.
    """
    if request.include_solution:
        output_requirements = """Provide:
1. The full problem statement.
2. A concise worked solution.
3. A clear final answer.
Start the problem with [[PROBLEM]].
Start the worked solution with [[SOLUTION]].
End the solution with the heading **Final Answer:** followed by the answer.
Never write [[FINAL ANSWER]] or place brackets around that heading."""
    else:
        output_requirements = """Provide only the full problem statement.
Do not include a solution, answer, hint, setup, or answer key.
End immediately after the problem statement."""

    if requested_context == "chemistry":
        return f"""
MANDATORY FINAL TASK

Create one original CHEMISTRY story problem about:
{request.subject}

The chemistry setting must be central to the mathematics.

Use single-variable calculus appropriate for MATH 1191–1210.

Do not use:
- products A and B,
- labor or material constraints,
- linear programming,
- a generic business-profit example,
- an unrelated abstract polynomial with no chemistry interpretation.

Possible chemistry settings include:
- concentration of a reacting substance,
- reaction yield,
- temperature of a chemical mixture,
- surface area or volume of a reaction vessel,
- chemical production cost,
- decay of a chemical concentration.

Student's exact request:
{request.other_details.strip()}

{output_requirements}

Use LaTeX delimiters around every mathematical expression.
""".strip()

    if requested_context:
        return f"""
MANDATORY FINAL TASK

Create one original {requested_context.upper()} application about:
{request.subject}

The requested context must be central to the problem or explanation.

Student's exact request:
{request.other_details.strip()}

Use calculus appropriate for MATH 1191–1210.
Do not replace the requested context with an unrelated generic example.

{output_requirements}
Use LaTeX delimiters around mathematical expressions.
""".strip()

    if request.request_type == "practice_problems":
        return f"""
MANDATORY FINAL TASK

{student_prompt}

Create one complete, original practice problem.

{output_requirements}

Use single-variable calculus appropriate for MATH 1191–1210.
Do not use linear programming unless the student explicitly asks for it.
Use LaTeX delimiters around every mathematical expression.
""".strip()

    if request.request_type == "quiz_me":
        return f"""
MANDATORY FINAL TASK

{student_prompt}

Ask exactly one quiz question.
Do not provide the answer yet.
Use LaTeX delimiters around every mathematical expression.
""".strip()

    return f"""
MANDATORY FINAL TASK

{student_prompt}

Answer the exact student request directly and completely.
Use calculus appropriate for MATH 1191–1210.
Use LaTeX delimiters around every mathematical expression.
""".strip()


# ── Generation endpoint ──

@app.post("/generate_prompt")
def generate_prompt(request: GeneratePrompt) -> dict[str, Any]:
    topic_names = [topic["name"] for topic in TOPICS]

    if request.subject not in topic_names:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid subject '{request.subject}'. "
                f"Must be one of: {topic_names}"
            ),
        )

    selected_topic = next(topic for topic in TOPICS if topic["name"] == request.subject)
    material_name = f"{selected_topic['id']}.json"

    if request.follow_up:
        follow_up_text = request.other_details.strip()
        if not follow_up_text:
            raise HTTPException(status_code=400, detail="Follow-up message cannot be empty")

        embedding = embed_query(f"{request.subject}\n{follow_up_text}")
        related_chunks = query_similar_chunks(
            embedding,
            min_score=PROMPT_SIMILARITY_THRESHOLD,
            material_name=material_name,
        )[:3]
        definitions = select_chunks_by_keywords(
            TOPIC_KEYWORDS.get(request.subject, []),
            kind="definition",
        )[:PROMPT_DEFINITION_COUNT]
        prompt_text = generate_follow_up_prompt(
            student_prompt=follow_up_text,
            conversation=[message.model_dump() for message in request.conversation],
            related_chunks=related_chunks,
            definitions=definitions,
        )
        print(
            f"Follow-up retrieved {len(related_chunks)} related chunk(s) and "
            f"{len(definitions)} definition chunk(s)."
        )
        raw_response = generate_qwen_response(prompt_text)
        return {
            "prompt": prompt_text,
            "response": _format_model_output(raw_response),
            "problem": None,
            "solution": None,
            "sources": _source_names(related_chunks, definitions),
        }

    student_prompt = _build_student_prompt(request)
    requested_context = _requested_context(request)

    print(f"Student prompt characters: {len(student_prompt)}")
    print(f"Requested context: {requested_context}")

    embedding = embed_query(student_prompt)

    # Generic examples can overpower explicit context requests.
    # When a context is requested, omit generic problem/example chunks.
    if requested_context is not None:
        practice_problems: list[dict[str, Any]] = []
        extra_contents: list[dict[str, Any]] = []
    else:
        practice_problems = query_similar_chunks(
            embedding,
            kind="problem",
            min_score=PROMPT_SIMILARITY_THRESHOLD,
            material_name=material_name,
        )[:PROMPT_PROBLEM_COUNT]

        extra_contents = query_similar_chunks(
            embedding,
            kind="other_material",
            min_score=PROMPT_SIMILARITY_THRESHOLD,
            material_name=material_name,
        )[:PROMPT_EXTRA_CONTENT_COUNT]

    definitions = select_chunks_by_keywords(
        TOPIC_KEYWORDS.get(request.subject, []),
        kind="definition",
    )[:PROMPT_DEFINITION_COUNT]

    print(
        "Retrieved "
        f"{len(practice_problems)} problem chunk(s), "
        f"{len(extra_contents)} extra-content chunk(s), and "
        f"{len(definitions)} definition chunk(s)."
    )

    rag_prompt = generate_prompt_internal(
        student_prompt=student_prompt,
        include_solution=request.include_solution,
        practice_problems=practice_problems,
        extra_contents=extra_contents,
        definitions=definitions,
    )

    final_directive = _build_context_directive(
        request=request,
        student_prompt=student_prompt,
        requested_context=requested_context,
    )

    # Place the mandatory instruction last so it survives left truncation.
    prompt_text = (
        f"{rag_prompt}\n\n"
        f"{'=' * 60}\n\n"
        f"{final_directive}"
    )

    print(f"Final RAG prompt characters: {len(prompt_text)}")
    print("Starting Qwen generation...")

    raw_response = generate_qwen_response(prompt_text)
    problem, solution = _split_problem_solution(raw_response)

    print("Qwen generation complete.")

    return {
        # Keep this while debugging. Remove it in production if the assembled
        # prompt exposes instructor-only course materials.
        "prompt": prompt_text,
        "response": _format_model_output(raw_response),
        "problem": problem,
        "solution": solution,
        "sources": _source_names(practice_problems, extra_contents, definitions),
    }


# ── Frontend and health routes ──

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
