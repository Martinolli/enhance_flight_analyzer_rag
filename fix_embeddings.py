#!/usr/bin/env python3
"""
Script to fix ChromaDB embedding dimension mismatch.
This will recreate the collection with the correct embedding dimensions.
"""

import os
import shutil
from pathlib import Path
import chromadb
from components.rag.ingest import ingest_documents

def main():
    db_path = ".ragdb"
    backup_path = ".ragdb_backup"
    knowledge_base_dir = "docs/knowledge_base"
    
    print(f"🔍 Checking current setup...")
    
    # Check if knowledge base exists
    if not Path(knowledge_base_dir).exists():
        print(f"❌ Knowledge base directory not found: {knowledge_base_dir}")
        print("Please ensure your documents are in the correct directory.")
        return
    
    # Check current collection
    if Path(db_path).exists():
        print(f"📁 Found existing database at {db_path}")
        try:
            client = chromadb.PersistentClient(path=db_path)
            coll = client.get_collection("flight_test_kb")
            count = coll.count()
            print(f"📊 Current collection has {count} documents")
            
            # Create backup
            if Path(backup_path).exists():
                shutil.rmtree(backup_path)
            shutil.copytree(db_path, backup_path)
            print(f"💾 Created backup at {backup_path}")
            
        except Exception as e:
            print(f"⚠️  Error accessing current database: {e}")
    
    # Remove old database
    if Path(db_path).exists():
        print(f"🗑️  Removing old database...")
        shutil.rmtree(db_path)
    
    # Recreate with correct embeddings
    print(f"🔄 Recreating database with correct embeddings...")
    print(f"📖 Ingesting documents from {knowledge_base_dir}...")
    
    try:
        ingest_documents(knowledge_base_dir, db_path, chunk_size=500)
        print(f"✅ Database recreated successfully!")
        
        # Verify new dimensions
        client = chromadb.PersistentClient(path=db_path)
        coll = client.get_collection("flight_test_kb")
        new_count = coll.count()
        print(f"📊 New collection has {new_count} documents")
        
        # Test embedding dimension
        from components.rag.ingest import embed_texts
        test_embedding = embed_texts(["test"])
        print(f"🧠 New embedding dimension: {len(test_embedding[0])}")
        
    except Exception as e:
        print(f"❌ Error recreating database: {e}")
        
        # Restore backup if available
        if Path(backup_path).exists():
            print(f"🔄 Restoring backup...")
            if Path(db_path).exists():
                shutil.rmtree(db_path)
            shutil.copytree(backup_path, db_path)
            print(f"✅ Backup restored")

if __name__ == "__main__":
    main()