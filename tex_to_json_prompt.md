# LaTeX Lesson → JSON Conversion Prompt

You are converting a LaTeX lesson **KEY** file (a calculus lesson with worked
solutions) into a single structured JSON problem-bank file. The JSON must match
the exact format of the existing files in `JSON/` (`lesson1.json` … `lesson4.json`).

You will be given:
- The path to one lesson `*_KEY.tex` file (authoritative for structure, math, and answers).
- Optionally, a directory of plain-text QA files (`Q:` / `A:` / `KEYWORDS:` format)
  produced from that lesson. When provided, use them as the source for the
  `*_plain` fields. When not provided, generate the plain text yourself from the
  LaTeX (see "Plain text rules" below).

Output **only** a JSON file. Do not explain your process in the file.

---

## 1. Study the examples first

Before converting, READ these source/result pairs. They define the target format.
Match them exactly.

| Source LaTeX | Resulting JSON | What it demonstrates |
|---|---|---|
| `MATH_1190___1210_Materials__Spring_2026_/Spring 2026/Lessons/Week1/lesson1_SlopesROC_KEY.tex` | `JSON/lesson1.json` | Basic problem + subproblems, plain/latex pair |
| `.../Week2/lesson2_IntroLimits_KEY.tex` | `JSON/lesson2.json` | `definitions`, table-based problems |
| `.../Week2/lesson3_LimitsContinuity_KEY.tex` | `JSON/lesson3.json` | `definitions` array (2 entries), grouped subproblems, two learning tags |
| `.../Week3/lesson4_LimitsInfinity_KEY_1190_S26.tex` | `JSON/lesson4.json` | `other_material` (mini-lecture), form-grouped problems |

The plain-text QA conventions are defined in `QA_data/ex_prompt`. Follow them for
every `*_plain` field.

---

## 2. Top-level JSON schema

```json
{
  "name": "<lesson title from \\TypeTitle, cleaned of LaTeX>",
  "week": <integer>,
  "learning_objectives": [
    {"tag": <int>, "description": "<one Intended Learning Outcome, cleaned>"}
  ],
  "contents": {
    "definitions": [
      {"term": "<name>", "definition_plain": "<...>", "definition_latex": "<...>"}
    ],
    "other_material": [
      {"type": "<mini-lecture|key-idea|summary|notation|discussion|theorem>",
       "content_plain": "<...>", "content_latex": "<...>"}
    ],
    "problems": [ <problem>, ... ]
  },
  "graph_dependent_excluded": [
    {"type": "<problem|definition|mini-lecture|discussion|example>",
     "name": "<short label>",
     "reason": "<why excluded — name the figure/graph it depends on>",
     "location": "<tex line range or section heading>"}
  ]
}
```

Rules:
- `definitions` is **always an array** (even with one entry). Omit the key entirely if the lesson has no definitions.
- `other_material` holds textual lecture content (see §5). Omit the key if there is none.
- `graph_dependent_excluded` is the **LAST** top-level key in the file (after `contents`). It is **required** — output it even if empty (`[]`). See §6.

### Problem object

```json
{
  "name": "<short descriptive name, e.g. \"Exercise 1\" or \"Form c/0: Example\">",
  "learning_tag": <int matching a learning_objectives tag>,
  "context_plain": "<shared stem/prompt, or \"\" if the question is self-contained>",
  "context_latex": "<same, LaTeX>",
  "keywords": ["<2-6 calculus keywords>"],
  "subproblems": [
    {
      "part": "a",
      "plain_text": {"question": "<...>", "answer": "<...>"},
      "latex":      {"question": "<...>", "answer": "<...>"}
    }
  ]
}
```

- `keywords` live at the **problem head** (not per-subproblem).
- Every problem has a `subproblems` array. A single-question problem gets **one**
  subproblem with `"part": "a"` and `context_*` set to `""`.
- When a single `\exercise`/`\example` has a shared prompt followed by several
  `\task`/`\item` parts, that is **ONE** problem; the shared prompt is the
  `context_*` and each part is a subproblem (`a`, `b`, `c`, …).

---

## 3. Metadata extraction

- `name`: from `\def\TypeTitle{...}` — strip `\\`, "Key", "Solutions", trailing whitespace.
- `week`: from the file's `WeekN` folder.
- `learning_objectives`: one entry per **Intended Learning Outcomes** bullet.
  `tag` numbers come from the `\target{Ln}` label (L1→1, L2→2, …). If several
  bullets share a target, you may merge them into one description for that tag.

---

## 4. Plain vs LaTeX rules

For every `*_plain` field:
- Strip all LaTeX. Convert math to readable ASCII per `QA_data/ex_prompt`
  (e.g. `\frac{a}{b}` → `a/b`, `\sqrt{x}` → `sqrt(x)`, `\lim_{x\to a}` → `lim as x -> a`,
  `\infty` → `infinity`, `$x^2$` → `x^2`).
- If QA plain files are supplied, copy their `Q:`/`A:` text **verbatim** into
  `question`/`answer`.

For every `*_latex` field:
- Keep the source math. Normalize custom macros to standard LaTeX:
  `\ds` → `\displaystyle`, `\limx[a]` → `\lim_{x\to a}`, `\dfrac` → `\frac`.
- You may drop layout-only macros (`\vs`, `\hs`, `\hfill`, `\mpageT`, `\itemC`,
  `\blue`, `\col`, `\unline` wrappers) but **keep the mathematical content** inside them.

General:
- **Preserve numbers, variable names, and source errors exactly.** Do not "fix" the math.
- **Tables (`tabular`/`array`) are NOT graphs.** Include table-based problems
  normally: render the table as an ASCII grid in `*_plain` and as a `tabular`
  environment in `*_latex` (see `lesson2.json`).

---

## 5. `other_material` (textual lecture content)

Capture mini-lectures, "Key Idea" boxes, summaries, notation discussions, and
**theorems** that are **textual and do not require a figure to understand**. Use
`type` ∈ `mini-lecture`, `key-idea`, `summary`, `notation`, `discussion`, `theorem`.

**Theorems** are statements named and asserted in the lesson (often in an
`\important`/`\theorem` box or bold inline), e.g.:

> `({\bf Extreme Value Theorem}) {\bf If} $f$ is {\em continuous} on $[a,b]$, {\bf then} $f$ attains an absolute maximum and minimum somewhere on $[a,b]$.`

Record these as `{"type": "theorem", ...}`. Lead the content with the theorem's
name, then its statement, e.g. `content_plain`: "Extreme Value Theorem: If f is
continuous on [a,b], then f attains an absolute maximum and minimum somewhere on
[a,b]." and `content_latex` keeping the math: "(\textbf{Extreme Value Theorem})
If $f$ is continuous on $[a,b]$, then $f$ attains an absolute maximum and minimum
somewhere on $[a,b]$." (A named *definition* still goes in `definitions`; a named
*theorem/result* goes here in `other_material`.)

Do **not** turn lecture content into fake problems. Do **not** include it if it
only makes sense alongside a graph (instead, log it per §6).

---

## 6. GRAPH HANDLING — read carefully

**Definition of graph-dependent.** An item is graph-dependent if its *question*
or its *answer/solution* cannot be understood or answered without a figure. Treat
something as graph-dependent if ANY of these hold:
- It contains a `tikzpicture` / `axis` / `\addplot` environment that is part of the prompt or answer.
- Its text refers to a figure: "the graph", "shown below/above/to the right",
  "the graph of …", "use the graph", "see graph", "sketch", "draw a function", "on the axes below".
- The answer is a label of a pictured graph (e.g. "graphs A, B, C") or requires reading values off a plot.

**What to do with graph-dependent items:**
1. **Do NOT** include them in `problems`, `definitions`, or `other_material`.
2. **DO** log every one of them in the top-level `graph_dependent_excluded`
   array at the **bottom** of the file. Each entry records `type`, a short
   `name`, a `reason` that names the figure it relies on, and the `location`
   (tex line range or nearest section heading).

**Mixed content** (a mini-lecture or problem that has BOTH usable text AND an
accompanying figure): include the text in the appropriate place
(`other_material` or `problems`) **and** add a `graph_dependent_excluded` entry
noting the figure was dropped, so nothing is silently lost.

**Not graphs:** `tabular`/`array` tables, inline math, and number lines written
as text. Include these normally.

> Note: the example files `lesson1`–`lesson4.json` excluded their graph problems
> *before* this logging rule existed, so some of them do not yet contain a
> populated `graph_dependent_excluded` array. You must still produce one.

---

## 7. Workflow

1. Read the `*_KEY.tex` file (and QA plain files, if provided).
2. Extract metadata (§3).
3. Walk the document top to bottom. Classify each block as: definition →
   `definitions`; textual lecture → `other_material`; solvable problem →
   `problems`; graph-dependent anything → `graph_dependent_excluded`.
4. Assemble the JSON with `graph_dependent_excluded` as the final top-level key.
5. Validate: `python3 -m json.tool <output>.json` must succeed.
6. Write the result to `JSON/<lessonN>.json`.

Output the JSON only.
