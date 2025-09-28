"""
# components/rag/ingest.py
Handles document ingestion and embedding for the RAG system.
This includes text extraction, chunking, embedding, and storage in ChromaDB.

Uses either OpenAI embeddings (if OPENAI_API_KEY is set) or local sentence-transformers.
How to use:
1. Ensure you have the required packages installed:
   - `chromadb`
   - `sentence-transformers`
   - `python-dotenv`
   - `pypdf` (optional, for PDF support)
2. Set the `OPENAI_API_KEY` environment variable if you want to use OpenAI embeddings.
3. Call `ingest_documents(source_dir, db_path, chunk_size)` to ingest documents from `source_dir` into ChromaDB at `db_path`.
4. Call `embed_texts(texts)` to embed the texts using the appropriate backend.

"""



import os
from typing import List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

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
        # Try better models for technical content, fallback to lighter ones
        model_options = [
            "sentence-transformers/all-MiniLM-L12-v2",  # Better than L6, still fast
            "sentence-transformers/all-mpnet-base-v2",   # High quality, slower
            "sentence-transformers/all-MiniLM-L6-v2"     # Fallback option
        ]
        
        for model_name in model_options:
            try:
                print(f"Loading embedding model: {model_name}")
                return SentenceTransformer(model_name)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
        
        raise RuntimeError("Could not load any embedding model")
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

def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Returns a list of embedding vectors for the input texts.
    Prefers OpenAI; falls back to sentence-transformers locally.
    Processes in batches for efficiency.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        _init_embedder()

    if not texts:
        return []

    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
        
        try:
            if _USE_OPENAI:
                # OpenAI path
                from openai import OpenAI
                client: OpenAI = _EMBEDDER
                # Force use of text-embedding-3-small to avoid dimension mismatch
                model = "text-embedding-3-small"  # 1536 dimensions
                resp = client.embeddings.create(model=model, input=batch)
                batch_embeddings = [d.embedding for d in resp.data]
            else:
                # Local sentence-transformers
                batch_embeddings = _EMBEDDER.encode(
                    batch, 
                    normalize_embeddings=True,
                    show_progress_bar=len(texts) > 10  # Show progress for large batches
                ).tolist()
            
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            # Add zero embeddings for failed batch to maintain alignment
            embedding_dim = 1536 if _USE_OPENAI else 384  # text-embedding-3-small: 1536, sentence-transformers: 384
            all_embeddings.extend([[0.0] * embedding_dim] * len(batch))
    
    return all_embeddings


def ingest_documents(source_dir: str, db_path: str = ".ragdb", chunk_size: int = 500):
    """
    Ingest documents from source_dir into ChromaDB at db_path.
    """
    import chromadb
    from pathlib import Path
    
    try:
        import pypdf
    except ImportError:
        print("Warning: pypdf not available. PDF parsing will be skipped.")
        pypdf = None
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("flight_test_kb")
    
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Process files
    documents = []
    metadatas = []
    ids = []
    
    doc_id = 0
    
    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            print(f"Processing: {file_path}")
            
            try:
                if file_path.suffix.lower() == '.pdf' and pypdf:
                    # Process PDF
                    text = extract_pdf_text(str(file_path))
                elif file_path.suffix.lower() in ['.txt', '.md']:
                    # Process text files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    print(f"Skipping unsupported file: {file_path}")
                    continue
                
                # Chunk the text
                chunks = chunk_text(text, chunk_size)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Skip empty chunks
                        documents.append(chunk.strip())
                        metadatas.append({
                            "source": str(file_path),
                            "chunk_id": i,
                            "file_type": file_path.suffix.lower()
                        })
                        ids.append(f"doc_{doc_id}_chunk_{i}")
                
                doc_id += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    if not documents:
        print("No documents found to ingest.")
        return
    
    print(f"Embedding {len(documents)} chunks...")
    
    # Generate embeddings
    embeddings = embed_texts(documents)
    
    # Add to collection
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Successfully ingested {len(documents)} chunks into {db_path}")


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        import pypdf
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 500, min_chunk_size: int = 50) -> List[str]:
    """Improved text chunking with better semantic boundaries."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    overlap = min(chunk_size // 4, 100)  # 25% overlap, max 100 chars
    start = 0
    
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If paragraph alone is too big, split it
        if len(paragraph) > chunk_size:
            # Save current chunk if it exists
            if current_chunk.strip() and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split large paragraph into sentences
            sentences = _split_into_sentences(paragraph)
            temp_chunk = ""
            
            for sentence in sentences:
                if len(temp_chunk + sentence) <= chunk_size:
                    temp_chunk += sentence + " "
                else:
                    if temp_chunk.strip() and len(temp_chunk) >= min_chunk_size:
                        chunks.append(temp_chunk.strip())
                    temp_chunk = sentence + " "
            
            if temp_chunk.strip() and len(temp_chunk) >= min_chunk_size:
                current_chunk = temp_chunk
        else:
            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk + "\n\n" + paragraph) > chunk_size:
                if current_chunk.strip() and len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
    
    # Add remaining chunk
    if current_chunk.strip() and len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences with better handling of technical content."""
    import re
    
    # Handle common abbreviations that shouldn't break sentences
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|vs|etc|Fig|Table|Eq)\.',
                  r'\1<DOT>', text)
    
    # Split on sentence endings
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Restore dots in abbreviations
    sentences = [s.replace('<DOT>', '.') for s in sentences if s.strip()]
    
    return sentences


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--source", required=True, help="Source directory containing documents")
    parser.add_argument("--db", default=".ragdb", help="ChromaDB database path")
    parser.add_argument("--chunk-size", type=int, default=500, help="Text chunk size")
    
    args = parser.parse_args()
    
    try:
        ingest_documents(args.source, args.db, args.chunk_size)
    except Exception as e:
        print(f"Error during ingestion: {e}")
        exit(1)