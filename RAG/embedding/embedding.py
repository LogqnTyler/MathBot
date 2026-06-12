from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")


def embed_text(text: str | list[str]) -> list[float]:
    if isinstance(text, str):
        text = text
    return model.encode(text).tolist()
