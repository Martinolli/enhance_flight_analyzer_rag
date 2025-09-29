"""
Bootstrap utilities to ensure a local Chroma DB (".ragdb") is available at runtime.

Priority order:
1) If an existing DB at db_path has documents, use it.
2) If RAG_DB_ZIP_URL is provided (via Streamlit secrets or env), download and extract.
3) If docs exist (docs/knowledge_base), ingest them to build the DB.

Use this on app start or before running retrieval to make cloud deploys robust.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Optional


def _get_secret(name: str) -> Optional[str]:
    """Best-effort read from Streamlit secrets, then environment."""
    try:
        import streamlit as st  # type: ignore

        try:
            val = st.secrets.get(name)  # type: ignore[attr-defined]
        except Exception:
            val = None
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(name)


def _log(msg: str):
    """Log to Streamlit if available, otherwise print."""
    try:
        import streamlit as st  # type: ignore

        st.info(msg)
    except Exception:
        print(msg)


def _db_has_documents(db_path: str) -> bool:
    try:
        import chromadb

        if not Path(db_path).exists():
            return False
        client = chromadb.PersistentClient(path=db_path)
        coll = client.get_or_create_collection("flight_test_kb")
        # If collection exists but empty, treat as not ready
        return (coll.count() or 0) > 0
    except Exception:
        return False


def _download_zip_to_memory(url: str) -> bytes:
    import requests

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    return resp.content


def _extract_zip_bytes(zip_bytes: bytes, dest_dir: str):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest)


def _ingest_from_docs(db_path: str, source_dir: str = "docs/knowledge_base", chunk_size: int = 500) -> bool:
    source = Path(source_dir)
    if not source.exists():
        return False
    _log(f"Building RAG DB from '{source_dir}' (this may take a few minutes on first run)...")
    try:
        from .ingest import ingest_documents

        ingest_documents(source_dir=source_dir, db_path=db_path, chunk_size=chunk_size)
        return _db_has_documents(db_path)
    except Exception as e:
        _log(f"RAG bootstrap: ingestion failed: {e}")
        return False


def ensure_rag_db(db_path: str = ".ragdb") -> bool:
    """
    Ensure a usable Chroma DB exists at db_path.

    Returns True if the DB is ready; False otherwise.
    """
    # 1) Already present and populated
    if _db_has_documents(db_path):
        return True

    # 2) Try fetching from a prebuilt zip
    zip_url = _get_secret("RAG_DB_ZIP_URL") or _get_secret("RAGDB_ZIP_URL")
    if zip_url:
        try:
            _log("Downloading prebuilt knowledge base (RAG_DB_ZIP_URL)...")
            data = _download_zip_to_memory(zip_url)
            _extract_zip_bytes(data, db_path)
            if _db_has_documents(db_path):
                _log("Knowledge base restored successfully.")
                return True
            else:
                _log("Downloaded KB did not contain documents. Falling back to local ingestion...")
        except Exception as e:
            _log(f"RAG bootstrap: download failed: {e}. Falling back to local ingestion...")

    # 3) Build from docs directory if available
    if _ingest_from_docs(db_path):
        return True

    # Not ready
    _log(
        "RAG DB is not available. Provide RAG_DB_ZIP_URL via secrets or include docs/knowledge_base and set OPENAI_API_KEY."
    )
    return False
