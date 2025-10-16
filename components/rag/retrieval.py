from typing import List, Dict, Any

import chromadb

from .ingest import embed_texts

COLLECTION_NAME = "flight_test_kb"


def _infer_collection_dimension(coll) -> int:
    """Best-effort detection of the stored embedding dimension."""
    try:
        sample = coll.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings") or []
        if embeddings and embeddings[0]:
            return len(embeddings[0])
    except Exception as exc:
        print(f"Warning: unable to inspect collection embedding dimension: {exc}")
    return 0

def get_collection(db_path: str):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(COLLECTION_NAME)

def retrieve(query: str, k: int = 6, db_path: str = ".ragdb") -> List[Dict[str, Any]]:
    coll = get_collection(db_path)
    target_dim = _infer_collection_dimension(coll) or None
    q_emb = embed_texts([query], target_dim=target_dim)[0]
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=max(1, k),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    docs: List[str] = res.get("documents", [[]])[0]
    metas: List[Dict] = res.get("metadatas", [[]])[0]
    dists: List[float] = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({"text": doc, "metadata": meta, "score": float(dist)})
    return out
