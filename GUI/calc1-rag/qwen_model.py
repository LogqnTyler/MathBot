from __future__ import annotations

import os
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = os.getenv(
    "QWEN_MODEL_NAME",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
)

MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "128"))

_tokenizer = None
_model = None
_model_lock = threading.Lock()


def load_qwen_model() -> None:
    """
    Load the tokenizer and model once when FastAPI starts.
    Optimized for Apple Silicon (MPS).
    """
    global _tokenizer, _model

    if _model is not None:
        return

    # Select the best available device
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

    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("Loading model...")

    # Load the entire model onto the selected device
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    _model.to(device)
    _model.eval()

    print(f"Model device: {next(_model.parameters()).device}")
    print(f"Model dtype: {next(_model.parameters()).dtype}")
def generate_qwen_response(
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response using Qwen2.5-Math-Instruct.
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Qwen model has not been loaded.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a mathematics tutor. Use the supplied course material "
                "when it is relevant. Explain the reasoning clearly and do not "
                "invent definitions, theorems, or course policies. Format "
                "mathematics using LaTeX."
            ),
        },
        {
            "role": "user",
            "content": prompt,
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
    ).to(_model.device)

    with _model_lock, torch.inference_mode():
        generated_ids = _model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            repetition_penalty=1.05,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Remove the original prompt tokens.
    new_tokens = generated_ids[:, model_inputs.input_ids.shape[1] :]

    response = _tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0]

    return response.strip()