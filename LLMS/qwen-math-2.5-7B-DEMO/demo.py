"""TIR demo using Qwen-Agent's TIRMathAgent against a local vLLM server.

TIRMathAgent handles the full tool-integrated-reasoning loop internally:
the model emits Python, the agent runs it, feeds the result back, and the
model finalizes in \\boxed{}. We just point it at the vLLM endpoint.
"""
import os
from pprint import pprint

from qwen_agent.agents import TIRMathAgent

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-Math-7B-Instruct")
PROBLEM = os.environ.get(
    "PROBLEM",
    r"Find the derivative of $f(x)=x^3\sin(x)$, then evaluate $f'(\pi)$.",
)

# Official TIR system prompt (from Qwen-Agent examples/tir_math.py).
TIR_SYSTEM = (
    "Please integrate natural language reasoning with programs to solve the "
    "problem above, and put your final answer within \\boxed{}."
)


def build_agent() -> TIRMathAgent:
    # Point at the local OpenAI-compatible vLLM server (no model_type ->
    # Qwen-Agent uses its OpenAI-compatible backend).
    llm_cfg = {
        "model": MODEL,
        "model_server": VLLM_URL,
        "api_key": "EMPTY",
        "generate_cfg": {"top_k": 1},  # deterministic
    }
    return TIRMathAgent(llm=llm_cfg, name="Qwen2.5-Math", system_message=TIR_SYSTEM)


def main() -> None:
    bot = build_agent()
    messages = [{"role": "user", "content": PROBLEM}]

    print(f"Problem: {PROBLEM}\n")

    # bot.run is a generator streaming partial results; the last yielded
    # value is the complete list of assistant/tool messages.
    response = []
    for response in bot.run(messages):
        pass

    print("=" * 60)
    pprint(response, indent=2)
    print("=" * 60)

    # Final assistant turn holds the concluding answer with \boxed{}.
    if response:
        print("\nFinal turn:\n" + (response[-1].get("content") or ""))


if __name__ == "__main__":
    main()
