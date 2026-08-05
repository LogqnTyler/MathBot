from qwen_model import load_qwen_model, generate_qwen_response


def main() -> None:
    print("Loading Qwen model...")
    load_qwen_model()
    print("Model loaded successfully.")

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if prompt.lower() in {"quit", "exit"}:
            print("Exiting.")
            break

        if not prompt:
            continue

        try:
            response = generate_qwen_response(prompt)
        except Exception as exc:
            print(f"\nGeneration failed: {exc}")
            continue

        print("\nResponse:\n")
        print(response)


if __name__ == "__main__":
    main()