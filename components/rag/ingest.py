import os
from typing import List, Optional

_EMBEDDER = None
_USE_OPENAI = False

def _try_openai():
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        client = OpenAI(api_key=api_key)
        # Quick sanity call (won't bill if not used here)
        return client
    except Exception:
        return None

def _get_local_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(
            "No embedding backend available. Set OPENAI_API_KEY or install sentence-transformers."
        ) from e

def _init_embedder():
    global _EMBEDDER, _USE_OPENAI
    client = _try_openai()
    if client:
        _USE_OPENAI = True
        _EMBEDDER = client
    else:
        _USE_OPENAI = False
        _EMBEDDER = _get_local_model()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns a list of embedding vectors for the input texts.
    Prefers OpenAI; falls back to sentence-transformers locally.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        _init_embedder()

    if _USE_OPENAI:
        # OpenAI path
        from openai import OpenAI
        client: OpenAI = _EMBEDDER
        # Use a cost-effective model; change if needed
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    else:
        # Local sentence-transformers
        return _EMBEDDER.encode(texts, normalize_embeddings=True).tolist()