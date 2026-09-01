

# MathBot

![MathBot calculus tutor interface](thumbnail.png)

MathBot is a RAG-powered chatbot designed to generate practice problems and solutions while drawing from course materials. We utilized LLMs to batch convert .tex files into .json files with lesson materials sorted into groups (questions and solutions, definitions, key concepts, summaries, etc) and later embedded with an open source BGE-M3 embedding model. We used a mix of vector and traditional relevance searches to retrieve the most relevant course material and produce practice problems tailed to student suggestions.

## Requirements

- Docker with Compose
- `uv`
- NVIDIA GPU with at least 8 GB VRAM

## Setup

Install locked Python dependencies:

```bash
uv sync
```

Start pgvector on `127.0.0.1:55432`:

```bash
docker compose up -d --wait
```

Import lesson JSON and create embeddings:

```bash
export DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:55432/postgres
uv run python data_ingestion/ingest_json.py
uv run python data_ingestion/embedding/main.py
```

Both ingestion commands are idempotent. First embedding run downloads
`BAAI/bge-m3` and can take several minutes.

## Website

Run from website directory so relative `JSON/` and `static/` paths resolve:

```bash
cd GUI/calc1-rag
DATABASE_URL=postgresql+pg8000://postgres:postgres@127.0.0.1:55432/postgres \
QWEN_MAX_INPUT_TOKENS=2048 \
QWEN_MAX_NEW_TOKENS=512 \
uv run uvicorn main:app --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080>. Startup downloads and loads
`Qwen/Qwen3-8B-AWQ` on CUDA plus BGE-M3 on CPU. Override model or token limits
with `QWEN_MODEL_NAME`, `QWEN_MAX_INPUT_TOKENS`, and
`QWEN_MAX_NEW_TOKENS`.

Health check:

```bash
curl http://127.0.0.1:8080/health
```
