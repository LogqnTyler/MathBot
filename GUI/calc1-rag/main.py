from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import uuid
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
    ensure_interactions_table,
    ensure_quiz_attempts_table,
    log_interaction,
    log_quiz_attempt,
    query_similar_chunks,
    query_similar_chunks_by_keywords,
    select_all_keywords,
    select_chunks_by_keywords,
)
from language_processing import (
    embed_query,
    generate_prompt_internal,
    generate_gemini_response,
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


def _parse_quiz_json(raw_response: str) -> dict[str, Any]:
    """Parse structured quiz JSON, tolerating harmless Markdown fences."""
    cleaned = raw_response.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)

    quiz_data = json.loads(cleaned)

    if not isinstance(quiz_data, dict):
        raise ValueError("Quiz response is not a JSON object.")

    return quiz_data


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

    with db_lifespan():
        ensure_interactions_table()
        ensure_quiz_attempts_table()
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
        "Follow any requested real-world context exactly. "
        "Include a complete problem statement and a concise worked solution."
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


class GeneratePrompt(BaseModel):
    subject: str
    request_type: RequestKind = "practice_problems"
    session_id: str = Field(
        ...,
        min_length=36,
        max_length=36,
        description="Pseudonymous browser-session UUID used to group interactions.",
    )
    other_details: str = Field(
        default="",
        min_length=0,
        max_length=500,
        description=(
            "The student's original question and any additional requirements "
            "that must be incorporated into the response."
        ),
    )


class FreeformQuestion(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The student's exact free-form question.",
    )
    session_id: str = Field(
        ...,
        min_length=36,
        max_length=36,
        description="Pseudonymous browser-session UUID used to group interactions.",
    )


def _infer_topics_from_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Infer course topics from semantically retrieved chunks.

    Each topic receives evidence from retrieved chunk keywords that overlap
    with TOPIC_KEYWORDS. Similarity scores weight stronger retrieval matches.
    Multiple topics are retained because a student question may legitimately
    span more than one course topic.
    """
    topic_scores: dict[str, float] = {}
    topic_matches: dict[str, set[str]] = {}

    for chunk in chunks:
        chunk_keywords = {
            str(keyword).lower()
            for keyword in (chunk.get("keywords") or [])
        }
        similarity = float(chunk.get("score") or 0.0)

        for topic, keywords in TOPIC_KEYWORDS.items():
            matches = {
                keyword
                for keyword in keywords
                if keyword.lower() in chunk_keywords
            }

            if not matches:
                continue

            topic_scores[topic] = topic_scores.get(topic, 0.0) + similarity
            topic_matches.setdefault(topic, set()).update(matches)

    ranked = sorted(
        topic_scores,
        key=lambda topic: topic_scores[topic],
        reverse=True,
    )

    return [
        {
            "topic": topic,
            "score": topic_scores[topic],
            "matched_keywords": sorted(topic_matches.get(topic, set())),
        }
        for topic in ranked
    ]


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

    The final directive is placed last so it has highest priority when the
    assembled prompt is long.
    """

    # Quiz behavior takes precedence over every requested application context.
    if request.request_type == "quiz_me":
        return f"""
MANDATORY FINAL TASK — QUIZ MODE

SELECTED COURSE TOPIC:
{request.subject}

STUDENT REQUEST:
{student_prompt}

Generate exactly ONE original multiple-choice quiz question that tests the
SELECTED COURSE TOPIC above.

TOPIC-SCOPE RULES:
- The selected topic is a HARD curricular boundary.
- Test a concept represented by the course definitions/material supplied above.
- Do NOT test an adjacent or later calculus topic merely because it uses
  similar vocabulary.
- The mathematical skill required to answer the question must itself belong
  to "{request.subject}".
- If the selected topic is "Limits and Continuity", test limits, one-sided
  limits, existence of limits, continuity, the continuity conditions, or the
  Continuity Theorem. Do NOT ask about derivatives, differentiability,
  tangent lines, difference quotients, or the limit definition of a derivative.
- Include only information that is relevant to answering the quiz question.
- Do not add an unnecessary formula, derivative, function, numerical value,
  or other detail that creates a side calculation unrelated to the skill being
  tested.
- If the question asks for a general theorem, definition, or necessary
  condition, do not add a specific example that could make the premise
  confusing or contradictory.

COURSE-WIDE MATHEMATICAL RESTRICTION:
- This course does NOT cover trigonometric functions.
- NEVER use sine, cosine, tangent, secant, cosecant, cotangent, inverse
  trigonometric functions, trigonometric identities, or trig-based examples.
- Do not use trigonometric functions even if they appear in retrieved course
  material.
- Use polynomial, rational, exponential, logarithmic, piecewise, or other
  non-trigonometric functions appropriate to the course instead.

MULTIPLE-CHOICE REQUIREMENTS:
- Provide exactly FOUR answer choices.
- Label them exactly A, B, C, and D.
- Exactly ONE choice must be correct.
- The three incorrect choices must be plausible mathematical distractors.
- Do not make the correct answer distinguishable by being longer, more
  detailed, or stylistically different from the distractors.
- The choices must test the intended mathematical concept rather than reading
  comprehension.
- Make sure the correct answer actually follows from the question as written.

OUTPUT FORMAT:
Return ONLY valid JSON. Do not use Markdown or a code fence.

The JSON must have exactly these fields:

{{
  "question": "The complete quiz question.",
  "options": {{
    "A": "First answer choice.",
    "B": "Second answer choice.",
    "C": "Third answer choice.",
    "D": "Fourth answer choice."
  }},
  "correct_answer": "A",
  "hint_1": "A conceptual hint that does not reveal the answer.",
  "hint_2": "A more specific hint that still does not directly reveal the answer.",
  "solution": "A concise worked solution explaining why the correct answer is correct."
}}

IMPORTANT:
- "correct_answer" must be exactly one of "A", "B", "C", or "D".
- The generated question, choices, correct answer, hints, and solution must all
  be mathematically consistent with one another.
- Do not generate a question whose answer depends on information not supplied
  in the question.
- Do not generate ambiguous questions with multiple defensible choices.
- Hints must help the student reason toward the answer rather than simply
  stating or paraphrasing the correct choice.
- The solution may identify the correct choice and explain the reasoning.
- The student should see only the question and four choices initially.

The student must select an answer before receiving feedback.
""".strip()

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

Provide:
1. A complete chemistry problem statement.
2. The function and its physically meaningful domain.
3. A concise worked calculus solution.
4. A clear interpretation of the optimum in the chemistry setting.

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

Answer completely and use LaTeX delimiters around mathematical expressions.
""".strip()

    if request.request_type == "practice_problems":
        return f"""
MANDATORY FINAL TASK — PRACTICE PROBLEM MODE

SELECTED COURSE TOPIC:
{request.subject}

STUDENT REQUEST:
{student_prompt}

Create exactly ONE complete, original practice problem that tests the
SELECTED COURSE TOPIC above.

TOPIC-SCOPE RULES:
- The selected topic is a HARD curricular boundary.
- The mathematical skill required to solve the problem must belong to
  "{request.subject}".
- Use the retrieved course definitions and material as curricular context.
- Do NOT switch to an adjacent or later calculus topic merely because it is
  mathematically related.
- Do NOT introduce a different topic as the main skill being tested.
- Follow any additional context requested by the student only if it can be
  used while remaining within the selected course topic.
- Include only information that is necessary or directly relevant to the
  skill being tested.
- Do not add distracting information that creates a separate calculation or
  apparent contradiction with the intended question.

OUTPUT FORMAT:
# Problem
Write the complete problem statement here.

# Solution
Give a concise worked solution here.

# Final Answer
Give the final answer here.

Use single-variable calculus appropriate for MATH 1191–1210.
Do not use linear programming unless the student explicitly asks for it.
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

    interaction_id = str(uuid.uuid4())

    # Preserve the exact text received from the browser for research logging.
    # Do not strip or otherwise normalize this value.
    original_prompt = request.other_details

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
        # Quiz questions should be grounded in course concepts without
        # exposing or imitating worked solutions from problem chunks.
        if request.request_type == "quiz_me":
            practice_problems = []
        else:
            practice_problems = query_similar_chunks_by_keywords(
                embedding,
                TOPIC_KEYWORDS.get(request.subject, []),
                kind="problem",
                min_score=PROMPT_SIMILARITY_THRESHOLD,
            )[:PROMPT_PROBLEM_COUNT]

        # Keep topic-selected generation inside the selected lesson.
        # Semantic similarity alone can cross lesson boundaries (for example,
        # a limits query retrieving derivative-definition material).
        extra_contents = select_chunks_by_keywords(
            TOPIC_KEYWORDS.get(request.subject, []),
            kind="other_material",
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
        practice_problems=practice_problems,
        extra_contents=extra_contents,
        definitions=definitions,
    )

    final_directive = _build_context_directive(
        request=request,
        student_prompt=student_prompt,
        requested_context=requested_context,
    )

    # Global course-scope rule applied to every generation mode.
    no_trig_rule = """
COURSE-WIDE CONTENT RESTRICTION:
MATH 1210 does NOT cover trigonometric functions.
- Do not introduce sine, cosine, tangent, secant, cosecant, cotangent,
  inverse trigonometric functions, or trigonometric identities.
- Do not create examples, practice problems, quiz questions, derivatives,
  integrals, limits, or applications involving trigonometric functions.
- Use polynomial, rational, exponential, logarithmic, piecewise, or other
  course-appropriate non-trigonometric functions instead.
- This restriction applies even if retrieved course material contains
  trigonometric examples.
""".strip()

    # Place the mandatory instruction and course restriction last so they
    # have highest priority in the assembled prompt.
    prompt_text = (
        f"{rag_prompt}\n\n"
        f"{'=' * 60}\n\n"
        f"{final_directive}\n\n"
        f"{no_trig_rule}"
    )

    print(f"Final RAG prompt characters: {len(prompt_text)}")
    print("Starting Qwen generation...")

    # Quiz mode uses structured JSON so the question, answer choices,
    # answer key, hints, and solution are generated together.
    if request.request_type == "quiz_me":
        quiz_schema = {
            "type": "OBJECT",
            "properties": {
                "question": {
                    "type": "STRING",
                },
                "options": {
                    "type": "OBJECT",
                    "properties": {
                        "A": {"type": "STRING"},
                        "B": {"type": "STRING"},
                        "C": {"type": "STRING"},
                        "D": {"type": "STRING"},
                    },
                    "required": ["A", "B", "C", "D"],
                },
                "correct_answer": {
                    "type": "STRING",
                    "enum": ["A", "B", "C", "D"],
                },
                "hint_1": {
                    "type": "STRING",
                },
                "hint_2": {
                    "type": "STRING",
                },
                "solution": {
                    "type": "STRING",
                },
            },
            "required": [
                "question",
                "options",
                "correct_answer",
                "hint_1",
                "hint_2",
                "solution",
            ],
        }

        raw_response = generate_gemini_response(
            prompt_text,
            response_schema=quiz_schema,
        )

        print("Quiz generation complete.")

        try:
            quiz_data = _parse_quiz_json(raw_response)
        except (json.JSONDecodeError, ValueError):
            print(
                "WARNING: Invalid quiz JSON on first attempt: "
                f"{raw_response[:2000]!r}"
            )
            print("Retrying quiz generation once...")

            raw_response = generate_gemini_response(
                prompt_text,
                response_schema=quiz_schema,
            )

            try:
                quiz_data = _parse_quiz_json(raw_response)
            except (json.JSONDecodeError, ValueError) as exc:
                print(
                    "ERROR: Invalid quiz JSON after retry: "
                    f"{raw_response[:2000]!r}"
                )
                raise HTTPException(
                    status_code=502,
                    detail="The quiz-generation model returned invalid structured output.",
                ) from exc

        if not isinstance(quiz_data, dict):
            raise HTTPException(
                status_code=502,
                detail="The quiz-generation model returned an invalid quiz object.",
            )

        required_fields = {
            "question",
            "options",
            "correct_answer",
            "hint_1",
            "hint_2",
            "solution",
        }

        if not required_fields.issubset(quiz_data):
            raise HTTPException(
                status_code=502,
                detail="The quiz-generation model returned an incomplete quiz.",
            )

        options = quiz_data["options"]

        if (
            not isinstance(options, dict)
            or set(options.keys()) != {"A", "B", "C", "D"}
            or any(not isinstance(options[key], str) for key in options)
        ):
            raise HTTPException(
                status_code=502,
                detail="The quiz-generation model returned invalid answer choices.",
            )

        if quiz_data["correct_answer"] not in {"A", "B", "C", "D"}:
            raise HTTPException(
                status_code=502,
                detail="The quiz-generation model returned an invalid answer key.",
            )

        formatted_response = quiz_data["question"]

    else:
        raw_response = generate_gemini_response(prompt_text)

        print("Qwen generation complete.")

        formatted_response = _format_model_output(raw_response)

    # Return the exact course chunks supplied to the generator so the
    # frontend can show students where MathBot's response was grounded.
    source_rows = practice_problems + extra_contents + definitions
    seen_source_ids: set[Any] = set()
    sources: list[dict[str, Any]] = []

    for row in source_rows:
        source_id = row.get("id")
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)

        sources.append(
            {
                "id": source_id,
                "kind": row.get("kind"),
                "topic": row.get("name") or row.get("kind") or "Course material",
                "text": (
                    row.get("content_latex")
                    or row.get("content_plain")
                    or row.get("Q_latex")
                    or row.get("Q_plain")
                    or ""
                ),
                "score": row.get("score"),
                "keywords": row.get("keywords") or [],
            }
        )

    # Record the completed interaction for later research/analysis.
    # original_prompt is the exact text received from the browser, while
    # student_prompt and prompt_text record the subsequent prompt construction.
    log_interaction(
        interaction_id=interaction_id,
        session_id=request.session_id,
        subject=request.subject,
        request_type=request.request_type,
        original_prompt=original_prompt,
        student_prompt=student_prompt,
        rag_prompt=prompt_text,
        retrieved_sources=[
            {
                "id": source.get("id"),
                "kind": source.get("kind"),
                "topic": source.get("topic"),
                "score": source.get("score"),
                "keywords": source.get("keywords", []),
            }
            for source in sources
        ],
        model_name="gemini-2.5-flash-lite",
        raw_model_response=raw_response,
        formatted_response=formatted_response,
    )

    result: dict[str, Any] = {
        "interaction_id": interaction_id,
        # Keep this while debugging. Remove it in production if the assembled
        # prompt exposes instructor-only course materials.
        "prompt": prompt_text,
        "response": formatted_response,
        "sources": sources,
    }

    if request.request_type == "quiz_me":
        result["quiz"] = quiz_data

    # Practice problems are returned in separate display fields so the
    # frontend can show the problem immediately while keeping the worked
    # solution hidden until the student requests it.
    if request.request_type == "practice_problems":
        import re

        problem_match = re.search(
            r"#\s*Problem\s*(.*?)(?=#\s*Solution|\Z)",
            formatted_response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        solution_match = re.search(
            r"#\s*Solution\s*(.*?)(?=#\s*Final Answer|\Z)",
            formatted_response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        answer_match = re.search(
            r"#\s*Final Answer\s*(.*)\Z",
            formatted_response,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if problem_match:
            result["problem"] = problem_match.group(1).strip()

            solution_parts = []
            if solution_match:
                solution_parts.append(solution_match.group(1).strip())
            if answer_match:
                solution_parts.append(
                    "**Final Answer:**\n\n" + answer_match.group(1).strip()
                )

            result["solution"] = "\n\n".join(solution_parts).strip()

    return result


# ── Direct free-form MathBot Q&A ──

@app.post("/ask")
def ask_mathbot(request: FreeformQuestion) -> dict[str, Any]:
    """
    Answer a student's free-form calculus question using course RAG.

    The exact browser-submitted question is preserved verbatim for research
    logging. Topic inference is performed from semantic retrieval evidence
    rather than requiring the student to choose a topic first.
    """
    interaction_id = str(uuid.uuid4())

    # Preserve exactly what arrived from the browser.
    original_prompt = request.question

    # Use the exact question for semantic retrieval.
    embedding = embed_query(original_prompt)

    retrieved = query_similar_chunks(
        embedding,
        min_score=PROMPT_SIMILARITY_THRESHOLD,
    )[:8]

    inferred_topics = _infer_topics_from_chunks(retrieved)
    primary_topic = (
        inferred_topics[0]["topic"]
        if inferred_topics
        else "Unclassified"
    )

    # Give the generator a small, semantically ranked set of course material.
    practice_problems = [
        row for row in retrieved if row.get("kind") == "problem"
    ][:PROMPT_PROBLEM_COUNT]

    extra_contents = [
        row for row in retrieved if row.get("kind") == "other_material"
    ][:PROMPT_EXTRA_CONTENT_COUNT]

    definitions = [
        row for row in retrieved if row.get("kind") == "definition"
    ][:PROMPT_DEFINITION_COUNT]

    rag_prompt = generate_prompt_internal(
        student_prompt=original_prompt,
        practice_problems=practice_problems,
        extra_contents=extra_contents,
        definitions=definitions,
    )

    final_directive = f"""
MANDATORY FINAL TASK — DIRECT QUESTION MODE

The student asked:

{original_prompt}

Answer the student's question directly.

RULES:
- Give a mathematically correct answer appropriate for introductory
  single-variable calculus.
- Use the retrieved course material as curricular context.
- Do not turn the request into a practice problem unless the student asks.
- Do not quiz the student unless the student asks.
- Explain the reasoning clearly and at an appropriate level.
- If the question spans multiple course topics, it is fine to connect them.
- Use LaTeX delimiters around mathematical expressions.
""".strip()

    no_trig_rule = """
COURSE-WIDE CONTENT RESTRICTION:
MathBot's course does NOT cover trigonometric functions.
- Do not introduce sine, cosine, tangent, secant, cosecant, cotangent,
  inverse trigonometric functions, or trigonometric identities.
- Do not create examples, exercises, derivatives, integrals, limits,
  or applications involving trigonometric functions.
- Use polynomial, rational, exponential, logarithmic, piecewise, or other
  course-appropriate non-trigonometric functions instead.
- This restriction applies even if retrieved course material contains
  trigonometric examples.
""".strip()

    prompt_text = (
        f"{rag_prompt}\n\n"
        f"{'=' * 60}\n\n"
        f"{final_directive}\n\n"
        f"{no_trig_rule}"
    )

    raw_response = generate_gemini_response(prompt_text)
    formatted_response = _format_model_output(raw_response)

    seen_source_ids: set[Any] = set()
    sources: list[dict[str, Any]] = []

    for row in retrieved:
        source_id = row.get("id")
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)

        sources.append(
            {
                "id": source_id,
                "kind": row.get("kind"),
                "topic": row.get("name") or row.get("kind") or "Course material",
                "text": (
                    row.get("content_latex")
                    or row.get("content_plain")
                    or row.get("Q_latex")
                    or row.get("Q_plain")
                    or ""
                ),
                "score": row.get("score"),
                "keywords": row.get("keywords") or [],
            }
        )

    # Keep the full topic-inference evidence with the retrieved-source record
    # so later analyses can reconstruct why a prompt was classified this way.
    research_sources = [
        {
            "id": source.get("id"),
            "kind": source.get("kind"),
            "topic": source.get("topic"),
            "score": source.get("score"),
            "keywords": source.get("keywords", []),
        }
        for source in sources
    ]

    research_sources.append(
        {
            "analysis_type": "topic_inference",
            "primary_topic": primary_topic,
            "inferred_topics": inferred_topics,
        }
    )

    log_interaction(
        interaction_id=interaction_id,
        session_id=request.session_id,
        subject=primary_topic,
        request_type="freeform",
        original_prompt=original_prompt,
        student_prompt=original_prompt,
        rag_prompt=prompt_text,
        retrieved_sources=research_sources,
        model_name="gemini-2.5-flash-lite",
        raw_model_response=raw_response,
        formatted_response=formatted_response,
    )

    return {
        "interaction_id": interaction_id,
        "response": formatted_response,
        "subject": primary_topic,
        "inferred_topics": inferred_topics,
        "sources": sources,
    }


# ── Frontend and health routes ──

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Three-attempt multiple-choice quiz tutoring ──

class CheckAnswerRequest(BaseModel):
    subject: str
    session_id: str = Field(min_length=36, max_length=36)
    question: str = Field(min_length=1, max_length=5000)
    student_answer: str = Field(min_length=1, max_length=1)
    correct_answer: str = Field(min_length=1, max_length=1)
    hint_1: str = Field(default="", max_length=5000)
    hint_2: str = Field(default="", max_length=5000)
    solution: str = Field(default="", max_length=10000)
    attempt: int = Field(ge=1, le=3)


@app.post("/check_answer")
def check_answer(request: CheckAnswerRequest) -> dict[str, Any]:
    """
    Deterministically check a multiple-choice answer.

    The correct answer is generated with the quiz and is compared directly
    with the student's selected choice. Gemini is not used to determine
    whether A, B, C, or D is correct.
    """

    student_answer = request.student_answer.strip().upper()
    correct_answer = request.correct_answer.strip().upper()

    if student_answer not in {"A", "B", "C", "D"}:
        raise HTTPException(
            status_code=400,
            detail="Student answer must be A, B, C, or D.",
        )

    if correct_answer not in {"A", "B", "C", "D"}:
        raise HTTPException(
            status_code=400,
            detail="Quiz answer key must be A, B, C, or D.",
        )

    correct = student_answer == correct_answer

    if correct:
        feedback = "Correct!"
        solution = None

        log_quiz_attempt(
            attempt_id=str(uuid.uuid4()),
            session_id=request.session_id,
            subject=request.subject,
            question=request.question,
            student_answer=student_answer,
            correct_answer=correct_answer,
            attempt=request.attempt,
            correct=True,
            feedback_shown=feedback,
        )

        return {
            "correct": True,
            "feedback": feedback,
            "solution": solution,
            "attempt": request.attempt,
            "attempts_remaining": 0,
        }

    if request.attempt == 1:
        feedback = request.hint_1 or "Take another look at the concept being tested."
        solution = None
    elif request.attempt == 2:
        feedback = request.hint_2 or "Think carefully about the definition or theorem being tested."
        solution = None
    else:
        feedback = "Let's work through it."
        solution = request.solution or "Review the question and the answer choices carefully."

    log_quiz_attempt(
        attempt_id=str(uuid.uuid4()),
        session_id=request.session_id,
        subject=request.subject,
        question=request.question,
        student_answer=student_answer,
        correct_answer=correct_answer,
        attempt=request.attempt,
        correct=False,
        feedback_shown=feedback,
    )

    return {
        "correct": False,
        "feedback": feedback,
        "solution": solution,
        "attempt": request.attempt,
        "attempts_remaining": max(0, 3 - request.attempt),
    }
