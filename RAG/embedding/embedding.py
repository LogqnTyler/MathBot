from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")


def embed_texts(text: str | list[str]) -> float:
    if isinstance(text, str):
        text = [text]
    return model.encode(text).tolist()
