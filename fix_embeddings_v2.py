#!/usr/bin/env python3
"""
Script to fix ChromaDB embedding dimension mismatch.
Creates a new collection with the correct embedding dimensions.
"""

import os
from pathlib import Path
import chromadb
from components.rag.ingest import ingest_documents

def main():
    db_path = ".ragdb"
    knowledge_base_dir = "docs/knowledge_base"
    
    print(f"🔍 Checking current setup...")
    
    # Check if knowledge base exists
    if not Path(knowledge_base_dir).exists():
        print(f"❌ Knowledge base directory not found: {knowledge_base_dir}")
        print("Please ensure your documents are in the correct directory.")
        return
    
    # Test current embedding dimensions
    from components.rag.ingest import embed_texts
    test_embedding = embed_texts(["test"])
    current_dim = len(test_embedding[0])
    print(f"🧠 Current embedding dimension: {current_dim}")
    
    # Create new collection with correct dimensions
    new_collection_name = "flight_test_kb_v2"
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        
        # Check if new collection already exists
        try:
            existing_coll = client.get_collection(new_collection_name)
            print(f"📁 Collection {new_collection_name} already exists with {existing_coll.count()} documents")
            
            # Test if dimensions match
            sample = existing_coll.peek(limit=1)
            if sample.get('embeddings') and len(sample['embeddings']) > 0:
                existing_dim = len(sample['embeddings'][0])
                if existing_dim == current_dim:
                    print(f"✅ Existing collection has correct dimensions ({existing_dim})")
                    return
                else:
                    print(f"❌ Existing collection has wrong dimensions ({existing_dim} vs {current_dim})")
                    print("Deleting and recreating...")
                    client.delete_collection(new_collection_name)
        except Exception:
            print(f"📝 Creating new collection: {new_collection_name}")
        
        # Create new collection and ingest documents
        collection = client.get_or_create_collection(new_collection_name)
        print(f"🔄 Ingesting documents from {knowledge_base_dir}...")
        
        # Modify ingest function to use new collection name temporarily
        import components.rag.ingest as ingest_module
        original_collection_name = ingest_module.COLLECTION_NAME if hasattr(ingest_module, 'COLLECTION_NAME') else "flight_test_kb"
        
        # Temporarily patch the collection name in ingest
        temp_ingest_file_content = f'''
def ingest_documents_temp(source_dir: str, db_path: str = ".ragdb", chunk_size: int = 500):
    """Temporary ingest function with correct collection name"""
    import chromadb
    from pathlib import Path
    from components.rag.ingest import embed_texts, extract_pdf_text, chunk_text
    
    try:
        import pypdf
    except ImportError:
        print("Warning: pypdf not available. PDF parsing will be skipped.")
        pypdf = None
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("{new_collection_name}")
    
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {{source_dir}}")
    
    # Process files
    documents = []
    metadatas = []
    ids = []
    
    doc_id = 0
    
    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            print(f"Processing: {{file_path}}")
            
            try:
                if file_path.suffix.lower() == '.pdf' and pypdf:
                    # Process PDF
                    text = extract_pdf_text(str(file_path))
                elif file_path.suffix.lower() in ['.txt', '.md']:
                    # Process text files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    print(f"Skipping unsupported file: {{file_path}}")
                    continue
                
                # Chunk the text
                chunks = chunk_text(text, chunk_size)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Skip empty chunks
                        documents.append(chunk)
                        metadatas.append({{
                            "source": str(file_path.relative_to(source_path)),
                            "chunk_id": i,
                            "file_type": file_path.suffix.lower()
                        }})
                        ids.append(f"{{file_path.stem}}_chunk_{{i}}")
                        doc_id += 1
                        
            except Exception as e:
                print(f"Error processing {{file_path}}: {{e}}")
                continue
    
    if not documents:
        print("No documents found to ingest!")
        return
    
    print(f"Embedding {{len(documents)}} document chunks...")
    embeddings = embed_texts(documents)
    
    print(f"Adding {{len(documents)}} documents to ChromaDB...")
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Successfully ingested {{len(documents)}} chunks from {{len(set(m['source'] for m in metadatas))}} files")
'''
        
        # Execute the temporary function
        exec(temp_ingest_file_content)
        locals()['ingest_documents_temp'](knowledge_base_dir, db_path)
        
        # Verify the new collection
        final_coll = client.get_collection(new_collection_name)
        final_count = final_coll.count()
        print(f"📊 New collection '{new_collection_name}' has {final_count} documents")
        print(f"🧠 Using embedding dimension: {current_dim}")
        print(f"✅ Database recreated successfully!")
        
        print(f"\n📝 Next steps:")
        print(f"1. Update your retrieval.py to use collection name: '{new_collection_name}'")
        print(f"2. Or rename the old collection and use the new one as default")
        
    except Exception as e:
        print(f"❌ Error creating new collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()