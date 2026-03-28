from dotenv import load_dotenv
import os
from config import EMBEDDING_MODEL

load_dotenv()

_EMBEDDINGS_CACHE = None


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    global _EMBEDDINGS_CACHE

    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    configured = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    model_candidates = [configured, EMBEDDING_MODEL, "BAAI/bge-base-en-v1.5"]
    deduped_candidates = []
    seen = set()
    for candidate in model_candidates:
        if candidate and candidate not in seen:
            deduped_candidates.append(candidate)
            seen.add(candidate)
    device = os.getenv("EMBEDDING_DEVICE", "cpu")

    failures = []
    for model_name in deduped_candidates:
        try:
            _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            _EMBEDDINGS_CACHE.embed_query("embedding-health-check")
            return _EMBEDDINGS_CACHE
        except Exception as exc:
            failures.append(f"{model_name}: {exc}")

    raise RuntimeError(
        "Failed to initialize local embeddings. Tried models: "
        + ", ".join(deduped_candidates)
        + ". Errors: "
        + " | ".join(failures)
    )
