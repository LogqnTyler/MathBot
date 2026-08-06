from __future__ import annotations

import gc
import os
import threading
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = os.getenv(
    "QWEN_MODEL_NAME",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
)

MAX_NEW_TOKENS = int(
    os.getenv("QWEN_MAX_NEW_TOKENS", "256")
)

MAX_INPUT_TOKENS = int(
    os.getenv("QWEN_MAX_INPUT_TOKENS", "160")
)

_tokenizer = None
_model = None
_model_lock = threading.Lock()


def load_qwen_model() -> None:
    """
    Load the tokenizer and model once when FastAPI starts.

    Uses Apple Metal (MPS) on Apple Silicon when available,
    CUDA when available, and CPU otherwise.
    """
    global _tokenizer, _model

    if _model is not None:
        return

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        print("Using Apple Metal (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("Using CPU")

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    _tokenizer.truncation_side = "left"
    print(f"Loading Qwen model ({MODEL_NAME})...")

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        trust_remote_code=True,
    )

    _model = _model.to(device)
    _model.eval()

    first_parameter = next(_model.parameters())

    print(f"Qwen model device: {first_parameter.device}")
    print(f"Qwen model dtype: {first_parameter.dtype}")


def generate_qwen_response(
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response using Qwen2.5-Math-Instruct.

    The input prompt is truncated to a fixed token limit to reduce
    memory use and slow prompt processing on an 8 GB Apple Silicon Mac.
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError(
            "Qwen model has not been loaded. "
            "Call load_qwen_model() first."
        )

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a mathematics tutor. Use the supplied course material "
                "when it is relevant. Follow all additional student requirements, "
                "including requests to use a particular application or context. "
                "Explain the reasoning clearly and do not invent definitions, "
                "theorems, or course policies. Format mathematics using LaTeX."
            ),
        },
        {
            "role": "user",
            "content": prompt.strip(),
        },
    ]

    formatted_prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = _tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(_model.device)

    input_token_count = model_inputs["input_ids"].shape[1]

    print(f"Qwen input tokens: {input_token_count}")
    print(f"Qwen maximum output tokens: {max_new_tokens}")

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "pad_token_id": _tokenizer.eos_token_id,
        "eos_token_id": _tokenizer.eos_token_id,
        "use_cache": True,
    }

    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()

    start = perf_counter()

    with _model_lock, torch.inference_mode():
        generated_ids = _model.generate(
            **model_inputs,
            **generation_kwargs,
        )

    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    elapsed = perf_counter() - start

    prompt_length = model_inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_length:]
    output_token_count = new_tokens.shape[1]

    response = _tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0].strip()

    print(
        f"Qwen generated {output_token_count} token(s) "
        f"in {elapsed:.1f} seconds."
    )

    del generated_ids
    del new_tokens
    del model_inputs

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return response
