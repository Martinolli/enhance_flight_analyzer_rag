"""
Hybrid Retrieval Module for Enhanced Flight Data Analyzer
Provides unified retrieval from both knowledge base documents and uploaded data.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.api.types import Documents, Embeddings, Metadatas

from .ingest import embed_texts
from .retrieval import get_collection as get_kb_collection


class HybridRetriever:
    """
    Unified retrieval from knowledge base and uploaded data.
    
    Supports:
    - Knowledge base document retrieval
    - Uploaded data retrieval (rows, columns, summaries)
    - Hybrid retrieval with result fusion
    - Metadata filtering
    - Re-ranking
    """
    
    # Collection names
    KB_COLLECTION = "flight_test_kb"
    DATA_ROWS_COLLECTION = "uploaded_data_rows"
    DATA_COLUMNS_COLLECTION = "uploaded_data_columns"
    DATA_SUMMARIES_COLLECTION = "uploaded_data_summaries"
    
    def __init__(self, db_path: str = ".ragdb"):
        """
        Initialize the hybrid retriever.
        
        Args:
            db_path: Path to ChromaDB database
        """
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collections
        try:
            self.kb_collection = self.client.get_collection(self.KB_COLLECTION)
        except Exception:
            self.kb_collection = None
        
        try:
            self.data_rows_collection = self.client.get_collection(self.DATA_ROWS_COLLECTION)
        except Exception:
            self.data_rows_collection = None
        
        try:
            self.data_columns_collection = self.client.get_collection(self.DATA_COLUMNS_COLLECTION)
        except Exception:
            self.data_columns_collection = None
        
        try:
            self.data_summaries_collection = self.client.get_collection(self.DATA_SUMMARIES_COLLECTION)
        except Exception:
            self.data_summaries_collection = None
    
    def retrieve_from_kb(
        self,
        query: str,
        k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from knowledge base documents.
        
        Args:
            query: Natural language query
            k: Number of results to return
            
        Returns:
            List of result dicts with keys: text, metadata, score, source_type
        """
        if self.kb_collection is None:
            return []
        
        try:
            # Generate query embedding
            q_emb = embed_texts([query])[0]
            
            # Query the collection
            results = self.kb_collection.query(
                query_embeddings=[q_emb],
                n_results=max(1, k),
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            out = []
            docs: List[str] = results.get("documents", [[]])[0]
            metas: List[Dict] = results.get("metadatas", [[]])[0]
            dists: List[float] = results.get("distances", [[]])[0]
            
            for doc, meta, dist in zip(docs, metas, dists):
                out.append({
                    "text": doc,
                    "metadata": meta,
                    "score": float(dist),
                    "source_type": "kb",
                    "source_name": meta.get("source", "Knowledge Base")
                })
            
            return out
            
        except Exception as e:
            print(f"Error retrieving from KB: {e}")
            return []
    
    def retrieve_from_data(
        self,
        query: str,
        k: int = 6,
        file_id: Optional[str] = None,
        embedding_type: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from uploaded data embeddings.
        
        Args:
            query: Natural language query
            k: Number of results to return
            file_id: Optional file ID filter
            embedding_type: "row", "column", "summary", or "all"
            
        Returns:
            List of result dicts with keys: text, metadata, score, source_type
        """
        results = []
        
        # Generate query embedding
        try:
            q_emb = embed_texts([query])[0]
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []
        
        # Determine which collections to query
        collections = []
        if embedding_type in ["row", "all"] and self.data_rows_collection:
            collections.append(("row", self.data_rows_collection))
        if embedding_type in ["column", "all"] and self.data_columns_collection:
            collections.append(("column", self.data_columns_collection))
        if embedding_type in ["summary", "all"] and self.data_summaries_collection:
            collections.append(("summary", self.data_summaries_collection))
        
        # Query each collection
        for emb_type, collection in collections:
            try:
                # Build where clause for filtering
                where = None
                if file_id:
                    where = {"file_id": file_id}
                
                # Query
                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=max(1, k),
                    where=where,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Format results
                docs: List[str] = res.get("documents", [[]])[0]
                metas: List[Dict] = res.get("metadatas", [[]])[0]
                dists: List[float] = res.get("distances", [[]])[0]
                
                for doc, meta, dist in zip(docs, metas, dists):
                    results.append({
                        "text": doc,
                        "metadata": meta,
                        "score": float(dist),
                        "source_type": "data",
                        "embedding_type": emb_type,
                        "source_name": meta.get("file_name", "Uploaded Data")
                    })
                
            except Exception as e:
                print(f"Error querying {emb_type} collection: {e}")
                continue
        
        # Sort by score (lower is better for distance)
        results.sort(key=lambda x: x["score"])
        
        return results[:k]
    
    def retrieve_hybrid(
        self,
        query: str,
        k: int = 10,
        sources: List[str] = ["kb", "data"],
        weights: Dict[str, float] = {"kb": 0.5, "data": 0.5},
        file_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from multiple sources and fuse results.
        
        Args:
            query: Natural language query
            k: Total number of results to return
            sources: List of sources to query ("kb", "data")
            weights: Weight for each source in fusion
            file_id: Optional file ID filter for data retrieval
            
        Returns:
            List of fused and ranked results
        """
        all_results = []
        
        # Retrieve from each source
        if "kb" in sources:
            kb_results = self.retrieve_from_kb(query, k=k)
            all_results.append(("kb", kb_results))
        
        if "data" in sources:
            data_results = self.retrieve_from_data(query, k=k, file_id=file_id)
            all_results.append(("data", data_results))
        
        # Fuse results using Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(all_results, weights=weights, k=60)
        
        return fused[:k]
    
    def filter_by_metadata(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to results.
        
        Args:
            results: List of result dicts
            filters: Dict of metadata filters
                Examples:
                - {"file_name": "flight_data.csv"}
                - {"source_type": "data"}
                - {"min_score": 0.7}
                
        Returns:
            Filtered results
        """
        filtered = []
        
        for result in results:
            match = True
            
            # Check each filter
            for key, value in filters.items():
                if key == "min_score":
                    if result.get("score", 1.0) > value:
                        match = False
                        break
                elif key == "max_score":
                    if result.get("score", 0.0) < value:
                        match = False
                        break
                elif key in result.get("metadata", {}):
                    if result["metadata"][key] != value:
                        match = False
                        break
                elif key in result:
                    if result[key] != value:
                        match = False
                        break
            
            if match:
                filtered.append(result)
        
        return filtered
    
    def rerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        strategy: str = "score"
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using different strategies.
        
        Args:
            results: List of result dicts
            query: Original query
            strategy: "score", "diversity", or "hybrid"
            
        Returns:
            Re-ranked results
        """
        if strategy == "score":
            # Simple score-based ranking (already done)
            return sorted(results, key=lambda x: x.get("score", 1.0))
        
        elif strategy == "diversity":
            # Maximize diversity (simple implementation)
            return self._diversify_results(results)
        
        elif strategy == "hybrid":
            # Combine score and diversity
            scored = sorted(results, key=lambda x: x.get("score", 1.0))
            return self._diversify_results(scored[:20])[:len(results)]
        
        return results
    
    # Private helper methods
    
    def _reciprocal_rank_fusion(
        self,
        results_lists: List[Tuple[str, List[Dict]]],
        weights: Dict[str, float],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.
        
        RRF score = sum(weight * 1 / (k + rank)) for each list
        
        Args:
            results_lists: List of (source_name, results) tuples
            weights: Weight for each source
            k: RRF constant (typically 60)
            
        Returns:
            Fused and ranked results
        """
        # Track scores for each unique result
        scores = defaultdict(float)
        result_map = {}  # Map result ID to result dict
        
        for source_name, results in results_lists:
            weight = weights.get(source_name, 1.0)
            
            for rank, result in enumerate(results, 1):
                # Create unique ID for this result
                result_id = self._create_result_id(result)
                
                # Calculate RRF score
                rrf_score = weight * (1.0 / (k + rank))
                scores[result_id] += rrf_score
                
                # Store result (first occurrence)
                if result_id not in result_map:
                    result_map[result_id] = result.copy()
                    result_map[result_id]["rrf_score"] = 0.0
                
                result_map[result_id]["rrf_score"] += rrf_score
        
        # Sort by RRF score (higher is better)
        ranked = sorted(
            result_map.values(),
            key=lambda x: x.get("rrf_score", 0.0),
            reverse=True
        )
        
        return ranked
    
    def _create_result_id(self, result: Dict[str, Any]) -> str:
        """Create unique ID for a result based on text and metadata."""
        text = result.get("text", "")
        source = result.get("source_type", "")
        # Use first 100 chars of text as ID
        return f"{source}_{hash(text[:100])}"
    
    def _diversify_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Diversify results to avoid redundancy.
        
        Simple implementation: alternate between different source types.
        """
        diversified = []
        by_source = defaultdict(list)
        
        # Group by source type
        for result in results:
            source = result.get("source_type", "unknown")
            by_source[source].append(result)
        
        # Alternate between sources
        max_len = max(len(v) for v in by_source.values()) if by_source else 0
        
        for i in range(max_len):
            for source in sorted(by_source.keys()):
                if i < len(by_source[source]):
                    diversified.append(by_source[source][i])
        
        return diversified


def retrieve_hybrid_simple(
    query: str,
    k: int = 10,
    db_path: str = ".ragdb",
    sources: List[str] = ["kb", "data"]
) -> List[Dict[str, Any]]:
    """
    Simple function interface for hybrid retrieval.
    
    Args:
        query: Natural language query
        k: Number of results
        db_path: Database path
        sources: Sources to query
        
    Returns:
        List of results
    """
    retriever = HybridRetriever(db_path=db_path)
    return retriever.retrieve_hybrid(query, k=k, sources=sources)

