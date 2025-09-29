# Providing the RAG database in deployments

Your Streamlit deploy doesn't contain `.ragdb/` by default (it's large and usually git-ignored). Use one of these options so retrieval works in the cloud:

## Option A — Prebuilt DB hosted as a zip (recommended)

1. Build the KB locally:

    - Put PDFs/txt in `docs/knowledge_base/`
    - Run ingestion (see RAG_SETUP.md) to create `.ragdb/`

2. Zip the DB:

    - `python tools/zip_ragdb.py` → creates `ragdb.zip`

3. Upload `ragdb.zip` to static hosting (e.g., GitHub Release asset, S3, GCS, Azure Blob with public read or signed URL).

4. Set a secret in your deployment:

    - Key: `RAG_DB_ZIP_URL`
    - Value: the direct URL to `ragdb.zip`

On app start, the page calls `ensure_rag_db()` which downloads and extracts the zip if the DB is missing.

Notes:

- The zip should contain the contents of `.ragdb/` directly (the tool already zips that way).
- The first run downloads once; subsequent runs reuse the persisted storage if your platform supports it.

## Option B — Build on startup from docs

If your deploy includes `docs/knowledge_base/` and has an embedding backend:

- Ensure `OPENAI_API_KEY` is set in secrets (or sentence-transformers is available)
- The app will ingest on first use via `ensure_rag_db()`
- This can take several minutes on cold start; prefer Option A for faster startup

## Option C — Bundle the DB with the image/artifact

If you build a container or custom artifact:

- Include the `.ragdb/` directory in the build output
- Make sure the runtime has read/write permissions for `.ragdb/`

## Streamlit secrets example

Place this in your `.streamlit/secrets.toml` locally (for testing) or in the platform’s secret manager:

```toml
OPENAI_API_KEY = "sk-..."
RAG_DB_ZIP_URL = "https://example.com/path/to/ragdb.zip"
```

The app also reads `OPENAI_API_KEY` and `RAG_DB_ZIP_URL` from environment variables if `st.secrets` is not available.

## Troubleshooting

- Retrieval returns no results:
  - Check that `.ragdb/` exists in the deployment filesystem and contains documents (`coll.count() > 0`).
  - If using a zip URL, verify it’s reachable without auth or uses a valid signed URL.
- Dimension mismatch errors:
  - Delete the DB and rebuild with the same embedding model used at query time (see RAG_SETUP.md).
- Cold-start is slow:
  - Prefer Option A to avoid rebuilding embeddings at runtime.
