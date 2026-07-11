# Qwen2.5-Math-7B TIR Demo (Docker)

Two services via Compose:
- `vllm` — serves `Qwen2.5-Math-7B-Instruct` on an OpenAI-compatible endpoint (needs the GPU).
- `demo` — runs Qwen-Agent's `TIRMathAgent`, which drives the full Tool-Integrated
  Reasoning loop (emit Python → execute → feed result back → finalize in `\boxed{}`)
  and executes the model's generated code **inside its own container**.

## Prerequisites
- NVIDIA GPU + driver (A6000 is plenty; 7B bf16 ≈ 15 GB VRAM).
- Docker with the NVIDIA Container Toolkit:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  ```

## Run
```bash
cd qwen-math-tir
docker compose up --build
```
First run downloads the weights (~15 GB) into your host HF cache; the `demo`
service waits on vLLM's healthcheck, so expect a pause before output appears.
You'll see the streamed agent turns (reasoning + code + tool output) and the
final `\boxed{}` answer.

## Try a different problem
No rebuild needed:
```bash
PROBLEM='Compute \int_0^1 x^2 e^x \, dx.' docker compose run --rm demo
```
Or edit the `PROBLEM` env in `docker-compose.yml`.

## Security note
Qwen-Agent's code interpreter is **not** itself sandboxed — it runs whatever
Python the model emits. Here the isolation comes from the `demo` container: it
has no host mounts and no GPU. Keep it that way, and don't repurpose this loop
for untrusted input in production without a real sandbox.

## Notes
- `generate_cfg={'top_k': 1}` for a deterministic demo.
- Weights persist in `~/.cache/huggingface` (override with `HF_HOME`); one-time download.
- On a shared workstation, lower `--gpu-memory-utilization` in `docker-compose.yml`.

## Next step toward your RAG tutor
Keep the `vllm` service as-is. In `demo.py`, retrieve course context with your
RAG pipeline and prepend it to the user message before `bot.run(...)`. The
agent's executed tool output is a free correctness signal — code error or a
symbolic mismatch means regenerate.
