"""
Data Ingestion Module for Enhanced Flight Data Analyzer
Handles ingestion of uploaded data files and generates embeddings for semantic search.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
import pandas as pd
from chromadb.api.types import Documents, Embeddings, Metadatas

from .rag.ingest import embed_texts


class DataIngestor:
    """
    Handles ingestion of uploaded data files into the vector database.
    
    Supports multiple embedding strategies:
    - Row-level: Embed individual data rows for finding similar records
    - Column-level: Embed column metadata for schema understanding
    - Summary-level: Embed dataset summaries for high-level queries
    - Hybrid: Combine all strategies (default)
    """
    
    # Collection names for different embedding types
    COLLECTION_ROWS = "uploaded_data_rows"
    COLLECTION_COLUMNS = "uploaded_data_columns"
    COLLECTION_SUMMARIES = "uploaded_data_summaries"
    
    # File size limit (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Maximum rows to embed (to control costs and performance)
    MAX_ROWS_TO_EMBED = 10000
    
    def __init__(self, db_path: str = ".ragdb"):
        """
        Initialize the data ingestor.
        
        Args:
            db_path: Path to ChromaDB database
        """
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collections
        self.rows_collection = self.client.get_or_create_collection(
            name=self.COLLECTION_ROWS,
            metadata={"description": "Row-level embeddings of uploaded data"}
        )
        self.columns_collection = self.client.get_or_create_collection(
            name=self.COLLECTION_COLUMNS,
            metadata={"description": "Column metadata embeddings"}
        )
        self.summaries_collection = self.client.get_or_create_collection(
            name=self.COLLECTION_SUMMARIES,
            metadata={"description": "Dataset summary embeddings"}
        )
    
    def ingest_file(
        self,
        file_path: str,
        file_name: str,
        embedding_strategy: str = "hybrid",
        max_rows: Optional[int] = None,
        sample_strategy: str = "uniform"
    ) -> Dict[str, Any]:
        """
        Ingest a data file and generate embeddings.
        
        Args:
            file_path: Path to the uploaded file
            file_name: Original filename
            embedding_strategy: "row", "column", "summary", or "hybrid"
            max_rows: Maximum rows to embed (default: MAX_ROWS_TO_EMBED)
            sample_strategy: "uniform", "random", or "stratified"
            
        Returns:
            Dict with ingestion results:
            {
                "file_id": str,
                "file_name": str,
                "row_count": int,
                "column_count": int,
                "rows_embedded": int,
                "columns_embedded": int,
                "summary_embedded": bool,
                "status": "success" | "error",
                "message": str
            }
        """
        try:
            # Validate file size
            file_size = os.path.getsize(file_path)
            if file_size > self.MAX_FILE_SIZE:
                return {
                    "status": "error",
                    "message": f"File size ({file_size / 1024**2:.1f} MB) exceeds limit ({self.MAX_FILE_SIZE / 1024**2:.1f} MB)"
                }
            
            # Read the file
            df = self.read_file(file_path)
            if df is None or df.empty:
                return {
                    "status": "error",
                    "message": "Failed to read file or file is empty"
                }
            
            # Generate unique file ID
            file_id = self._generate_file_id(file_name, df)
            
            # Check if file already ingested
            existing = self._get_file_metadata(file_id)
            if existing:
                return {
                    "status": "error",
                    "message": f"File already ingested (ID: {file_id}). Delete first if you want to re-ingest."
                }
            
            max_rows = max_rows or self.MAX_ROWS_TO_EMBED
            
            result = {
                "file_id": file_id,
                "file_name": file_name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "rows_embedded": 0,
                "columns_embedded": 0,
                "summary_embedded": False,
                "status": "success",
                "message": "Ingestion completed successfully"
            }
            
            # Execute embedding strategies
            if embedding_strategy in ["row", "hybrid"]:
                rows_embedded = self.generate_row_embeddings(
                    df, file_id, file_name, max_rows, sample_strategy
                )
                result["rows_embedded"] = rows_embedded
            
            if embedding_strategy in ["column", "hybrid"]:
                columns_embedded = self.generate_column_embeddings(
                    df, file_id, file_name
                )
                result["columns_embedded"] = columns_embedded
            
            if embedding_strategy in ["summary", "hybrid"]:
                summary_embedded = self.generate_summary_embedding(
                    df, file_id, file_name
                )
                result["summary_embedded"] = summary_embedded
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Ingestion failed: {str(e)}"
            }
    
    def read_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Read file based on extension.
        
        Supports: CSV, Excel (.xlsx, .xls), Parquet
        
        Args:
            file_path: Path to the file
            
        Returns:
            DataFrame or None if reading fails
        """
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == ".csv":
                # Try different encodings and separators
                for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        return df
                    except UnicodeDecodeError:
                        continue
                return None
            
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                return df
            
            elif ext == ".parquet":
                df = pd.read_parquet(file_path)
                return df
            
            else:
                print(f"Unsupported file format: {ext}")
                return None
                
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def generate_row_embeddings(
        self,
        df: pd.DataFrame,
        file_id: str,
        file_name: str,
        max_rows: int = 10000,
        sample_strategy: str = "uniform"
    ) -> int:
        """
        Generate embeddings for data rows.
        
        Args:
            df: DataFrame to embed
            file_id: Unique file identifier
            file_name: Original filename
            max_rows: Maximum rows to embed
            sample_strategy: Sampling strategy if df has more than max_rows
            
        Returns:
            Number of rows embedded
        """
        try:
            # Sample if necessary
            if len(df) > max_rows:
                if sample_strategy == "uniform":
                    # Take evenly spaced rows
                    indices = np.linspace(0, len(df) - 1, max_rows, dtype=int)
                    df_sample = df.iloc[indices].copy()
                elif sample_strategy == "random":
                    df_sample = df.sample(n=max_rows, random_state=42)
                else:  # stratified or default to random
                    df_sample = df.sample(n=max_rows, random_state=42)
            else:
                df_sample = df.copy()
            
            # Create text representations for each row
            row_texts = []
            row_metadatas = []
            row_ids = []
            
            for idx, row in df_sample.iterrows():
                text = self._create_row_text(row, df.columns.tolist())
                
                metadata = {
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_type": os.path.splitext(file_name)[1].lower().replace(".", ""),
                    "upload_timestamp": datetime.utcnow().isoformat(),
                    "row_index": int(idx),
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "embedding_type": "row",
                    "source_type": "data"
                }
                
                row_id = f"{file_id}_row_{idx}"
                
                row_texts.append(text)
                row_metadatas.append(metadata)
                row_ids.append(row_id)
            
            # Generate embeddings in batches
            batch_size = 100
            for i in range(0, len(row_texts), batch_size):
                batch_texts = row_texts[i:i+batch_size]
                batch_metadatas = row_metadatas[i:i+batch_size]
                batch_ids = row_ids[i:i+batch_size]
                
                # Generate embeddings
                embeddings = embed_texts(batch_texts)
                
                # Store in ChromaDB
                self.rows_collection.add(
                    documents=batch_texts,
                    embeddings=embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            return len(row_texts)
            
        except Exception as e:
            print(f"Error generating row embeddings: {e}")
            return 0
    
    def generate_column_embeddings(
        self,
        df: pd.DataFrame,
        file_id: str,
        file_name: str
    ) -> int:
        """
        Generate embeddings for column metadata.
        
        Args:
            df: DataFrame
            file_id: Unique file identifier
            file_name: Original filename
            
        Returns:
            Number of columns embedded
        """
        try:
            column_texts = []
            column_metadatas = []
            column_ids = []
            
            for col in df.columns:
                text = self._create_column_text(df, col)
                
                # Compute statistics for numeric columns
                stats = {}
                if pd.api.types.is_numeric_dtype(df[col]):
                    series = pd.to_numeric(df[col], errors="coerce")
                    stats = {
                        "mean": float(series.mean()) if not series.isna().all() else None,
                        "std": float(series.std()) if not series.isna().all() else None,
                        "min": float(series.min()) if not series.isna().all() else None,
                        "max": float(series.max()) if not series.isna().all() else None,
                        "median": float(series.median()) if not series.isna().all() else None
                    }
                
                metadata = {
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_type": os.path.splitext(file_name)[1].lower().replace(".", ""),
                    "upload_timestamp": datetime.utcnow().isoformat(),
                    "column_name": col,
                    "column_dtype": str(df[col].dtype),
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "embedding_type": "column",
                    "source_type": "data",
                    "statistics": json.dumps(stats) if stats else None
                }
                
                column_id = f"{file_id}_col_{col}"
                
                column_texts.append(text)
                column_metadatas.append(metadata)
                column_ids.append(column_id)
            
            # Generate embeddings
            embeddings = embed_texts(column_texts)
            
            # Store in ChromaDB
            self.columns_collection.add(
                documents=column_texts,
                embeddings=embeddings,
                metadatas=column_metadatas,
                ids=column_ids
            )
            
            return len(column_texts)
            
        except Exception as e:
            print(f"Error generating column embeddings: {e}")
            return 0
    
    def generate_summary_embedding(
        self,
        df: pd.DataFrame,
        file_id: str,
        file_name: str
    ) -> bool:
        """
        Generate embedding for dataset summary.
        
        Args:
            df: DataFrame
            file_id: Unique file identifier
            file_name: Original filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            text = self._create_summary_text(df, file_name)
            
            # Compute overall statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
            
            metadata = {
                "file_id": file_id,
                "file_name": file_name,
                "file_type": os.path.splitext(file_name)[1].lower().replace(".", ""),
                "upload_timestamp": datetime.utcnow().isoformat(),
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "datetime_columns": len(datetime_cols),
                "missing_percentage": float(df.isnull().sum().sum() / df.size * 100),
                "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
                "embedding_type": "summary",
                "source_type": "data"
            }
            
            # Generate embedding
            embeddings = embed_texts([text])
            
            # Store in ChromaDB
            self.summaries_collection.add(
                documents=[text],
                embeddings=embeddings,
                metadatas=[metadata],
                ids=[f"{file_id}_summary"]
            )
            
            return True
            
        except Exception as e:
            print(f"Error generating summary embedding: {e}")
            return False
    
    def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for an ingested file.
        
        Args:
            file_id: File identifier
            
        Returns:
            Metadata dict or None if not found
        """
        return self._get_file_metadata(file_id)
    
    def list_ingested_files(self) -> List[Dict[str, Any]]:
        """
        List all ingested files with metadata.
        
        Returns:
            List of file metadata dicts
        """
        try:
            # Get all summaries (one per file)
            results = self.summaries_collection.get()
            
            files = []
            if results and results.get("metadatas"):
                for metadata in results["metadatas"]:
                    files.append({
                        "file_id": metadata.get("file_id"),
                        "file_name": metadata.get("file_name"),
                        "file_type": metadata.get("file_type"),
                        "upload_timestamp": metadata.get("upload_timestamp"),
                        "total_rows": metadata.get("total_rows"),
                        "total_columns": metadata.get("total_columns"),
                        "numeric_columns": metadata.get("numeric_columns"),
                        "categorical_columns": metadata.get("categorical_columns"),
                        "missing_percentage": metadata.get("missing_percentage"),
                        "memory_usage_mb": metadata.get("memory_usage_mb")
                    })
            
            return files
            
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete all embeddings for a file.
        
        Args:
            file_id: File identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from all collections
            for collection in [self.rows_collection, self.columns_collection, self.summaries_collection]:
                # Get all IDs for this file
                results = collection.get(where={"file_id": file_id})
                if results and results.get("ids"):
                    collection.delete(ids=results["ids"])
            
            return True
            
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    # Private helper methods
    
    def _generate_file_id(self, file_name: str, df: pd.DataFrame) -> str:
        """Generate unique file ID based on filename and content hash."""
        # Create a hash of filename + shape + first few rows
        content = f"{file_name}_{df.shape}_{df.head(5).to_json()}"
        hash_obj = hashlib.md5(content.encode())
        return f"file_{hash_obj.hexdigest()[:16]}"
    
    def _get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata from summaries collection."""
        try:
            results = self.summaries_collection.get(
                ids=[f"{file_id}_summary"]
            )
            if results and results.get("metadatas"):
                return results["metadatas"][0]
            return None
        except Exception:
            return None
    
    def _create_row_text(self, row: pd.Series, columns: List[str]) -> str:
        """
        Create text representation of a row for embedding.
        
        Example output:
        "Timestamp: 2024-01-15 10:30:00, Altitude: 5000, Speed: 250, Temperature: 15"
        """
        parts = []
        for col in columns:
            value = row[col]
            if pd.notna(value):
                # Format based on type
                if isinstance(value, (int, np.integer)):
                    parts.append(f"{col}: {int(value)}")
                elif isinstance(value, (float, np.floating)):
                    parts.append(f"{col}: {float(value):.2f}")
                else:
                    parts.append(f"{col}: {value}")
        
        return ", ".join(parts)
    
    def _create_column_text(self, df: pd.DataFrame, column: str) -> str:
        """
        Create text representation of a column for embedding.
        
        Example output:
        "Column: Altitude, Type: float64, Range: 0 to 10000, Mean: 5000.00, 
         Std: 1500.00, Sample values: [1000, 2000, 3000, 4000, 5000]"
        """
        parts = [f"Column name: {column}"]
        parts.append(f"Data type: {df[column].dtype}")
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            series = pd.to_numeric(df[column], errors="coerce")
            if not series.isna().all():
                parts.append(f"Range: {series.min():.2f} to {series.max():.2f}")
                parts.append(f"Mean: {series.mean():.2f}")
                parts.append(f"Standard deviation: {series.std():.2f}")
                parts.append(f"Median: {series.median():.2f}")
        
        # Add sample values
        sample_size = min(5, len(df))
        samples = df[column].dropna().head(sample_size).tolist()
        if samples:
            parts.append(f"Sample values: {samples}")
        
        # Add missing value info
        missing_pct = df[column].isnull().sum() / len(df) * 100
        if missing_pct > 0:
            parts.append(f"Missing values: {missing_pct:.1f}%")
        
        return ", ".join(parts)
    
    def _create_summary_text(self, df: pd.DataFrame, file_name: str) -> str:
        """
        Create text representation of dataset summary.
        
        Example output:
        "Flight data file: flight_data_2024.csv. Contains 10000 rows and 25 columns.
         Numeric columns (15): Altitude, Speed, Temperature, ...
         Time range: 2024-01-15 to 2024-01-20.
         Data completeness: 98.5%"
        """
        parts = [f"Data file: {file_name}"]
        parts.append(f"Contains {len(df):,} rows and {len(df.columns)} columns")
        
        # Column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        
        if numeric_cols:
            col_list = ", ".join(numeric_cols[:10])
            if len(numeric_cols) > 10:
                col_list += f", and {len(numeric_cols) - 10} more"
            parts.append(f"Numeric columns ({len(numeric_cols)}): {col_list}")
        
        if categorical_cols:
            col_list = ", ".join(categorical_cols[:5])
            if len(categorical_cols) > 5:
                col_list += f", and {len(categorical_cols) - 5} more"
            parts.append(f"Categorical columns ({len(categorical_cols)}): {col_list}")
        
        if datetime_cols:
            parts.append(f"Datetime columns ({len(datetime_cols)}): {', '.join(datetime_cols)}")
            # Try to get time range
            for col in datetime_cols:
                try:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    parts.append(f"Time range in {col}: {min_date} to {max_date}")
                    break  # Just use first datetime column
                except Exception:
                    pass
        
        # Data quality
        completeness = (1 - df.isnull().sum().sum() / df.size) * 100
        parts.append(f"Data completeness: {completeness:.1f}%")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        parts.append(f"Memory usage: {memory_mb:.1f} MB")
        
        return ". ".join(parts) + "."

