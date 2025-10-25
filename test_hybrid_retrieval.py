"""
Test script for hybrid retrieval system.
"""

import sys
sys.path.insert(0, 'enhance_flight_analyzer_rag')

from components.rag.hybrid_retrieval import HybridRetriever

def main():
    print("=" * 80)
    print("Hybrid Retrieval System Test")
    print("=" * 80)
    
    # Initialize retriever
    print("\n1. Initializing HybridRetriever...")
    retriever = HybridRetriever(db_path=".ragdb")
    print("✓ HybridRetriever initialized")
    
    # Check available collections
    print("\n2. Checking available collections...")
    print(f"  KB collection: {'✓' if retriever.kb_collection else '✗'}")
    print(f"  Data rows collection: {'✓' if retriever.data_rows_collection else '✗'}")
    print(f"  Data columns collection: {'✓' if retriever.data_columns_collection else '✗'}")
    print(f"  Data summaries collection: {'✓' if retriever.data_summaries_collection else '✗'}")
    
    # Test data retrieval
    if retriever.data_summaries_collection:
        print("\n3. Testing data retrieval...")
        
        # Test queries
        test_queries = [
            "flight data with roll angle and pitch angle",
            "engine torque and temperature data",
            "accelerometer measurements",
            "AHRS attitude data"
        ]
        
        for query in test_queries:
            print(f"\n  Query: '{query}'")
            results = retriever.retrieve_from_data(query, k=3, embedding_type="summary")
            
            if results:
                print(f"  ✓ Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"    {i}. [{result['source_name']}] Score: {result['score']:.4f}")
                    print(f"       {result['text'][:150]}...")
            else:
                print("  ✗ No results found")
    
    # Test hybrid retrieval
    print("\n4. Testing hybrid retrieval...")
    
    test_query = "What are the key flight parameters in the data?"
    print(f"  Query: '{test_query}'")
    
    results = retriever.retrieve_hybrid(
        query=test_query,
        k=5,
        sources=["data"],  # Only data for now (KB might be empty)
        weights={"data": 1.0}
    )
    
    if results:
        print(f"  ✓ Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"    {i}. [{result.get('source_type', 'unknown')}] Score: {result.get('score', 0):.4f}")
            if 'rrf_score' in result:
                print(f"       RRF Score: {result['rrf_score']:.4f}")
            print(f"       {result['text'][:100]}...")
    else:
        print("  ✗ No results found")
    
    # Test metadata filtering
    print("\n5. Testing metadata filtering...")
    
    if results:
        filtered = retriever.filter_by_metadata(
            results=results,
            filters={"source_type": "data"}
        )
        print(f"  ✓ Filtered to {len(filtered)} data results")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
