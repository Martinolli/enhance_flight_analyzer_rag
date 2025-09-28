# RAG (Retrieval-Augmented Generation) Setup

## 1) Install dependencies

Add these to your environment (or requirements.txt):

```
chromadb
pypdf
openai
sentence-transformers
```

Optional:
- `tiktoken` (for token-aware chunking, not required here)

## 2) Prepare your corpus

Place PDFs/TXT in a folder, e.g.:

```
docs/knowledge_base/
  - Flight_Test_Handbook.pdf
  - Engine_Testing_Principles.pdf
  - notes_on_vibration.txt
```

Ensure you comply with licenses and fair-use.

## 3) Build the vector database

Run the ingestion script:

```bash
python -m components.rag.ingest --source docs/knowledge_base --db .ragdb
```

This creates a persistent Chroma database at `.ragdb/`.

## 4) Configure keys

- For OpenAI embeddings/LLM: set `OPENAI_API_KEY` in your environment or `st.secrets`.
- If no key is set, the app will fall back to local `sentence-transformers` for embeddings.
- The chat/report generation currently uses OpenAI Chat Completions (`OPENAI_API_KEY` required for that part).

## 5) Use the assistant

- Start the app and open the “Knowledge & Report Assistant” page.
- Ask questions (Q&A) or generate a report using your uploaded dataset from the main page.

## Notes

- Chroma database is local. You can commit it if you want collaborators to reuse it, or add `.ragdb/` to `.gitignore`.
- If you prefer FAISS or pgvector, the retrieval layer can be swapped by re-implementing `components/rag/retrieval.py`.