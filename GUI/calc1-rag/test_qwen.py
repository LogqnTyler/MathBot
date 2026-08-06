from time import perf_counter

from qwen_model import load_qwen_model, generate_qwen_response


print("Loading model...")

start = perf_counter()
load_qwen_model()
print(f"Model loaded in {perf_counter() - start:.1f}s")

prompt = "What is the difference between a local minimum and an absolute minimum?"

print("\nGenerating...")

start = perf_counter()
response = generate_qwen_response(
    prompt,
    max_new_tokens=64,
)

print(f"\nGenerated in {perf_counter() - start:.1f}s")
print("\nResponse:\n")
print(response)