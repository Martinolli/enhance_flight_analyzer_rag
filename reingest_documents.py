#!/usr/bin/env python3
"""
Simple script to re-ingest documents after deleting the old database.
"""

from components.rag.ingest import ingest_documents
from pathlib import Path

def main():
    db_path = ".ragdb"
    knowledge_base_dir = "docs/knowledge_base"
    
    # Check if old database still exists
    if Path(db_path).exists():
        print(f"❌ Old database still exists at {db_path}")
        print("Please delete the .ragdb folder first, then run this script again.")
        return
    
    # Check if knowledge base exists
    if not Path(knowledge_base_dir).exists():
        print(f"❌ Knowledge base directory not found: {knowledge_base_dir}")
        print("Please ensure your documents are in the docs/knowledge_base directory.")
        return
    
    # Check current embedding setup
    from components.rag.ingest import embed_texts
    test_embedding = embed_texts(["test"])
    current_dim = len(test_embedding[0])
    print(f"🧠 Current embedding dimension: {current_dim}")
    print(f"📖 Ingesting documents from {knowledge_base_dir}...")
    
    try:
        ingest_documents(knowledge_base_dir, db_path, chunk_size=500)
        print(f"✅ Documents ingested successfully!")
        
        # Verify the new collection
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        coll = client.get_collection("flight_test_kb")
        count = coll.count()
        print(f"📊 New collection has {count} documents")
        print(f"🎉 Ready to use with {current_dim}-dimensional embeddings!")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()