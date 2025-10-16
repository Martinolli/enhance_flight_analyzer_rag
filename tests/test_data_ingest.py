"""
Unit tests for the data ingestion module.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.data_ingest import DataIngestor


class TestDataIngestor(unittest.TestCase):
    """Test cases for DataIngestor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create temporary directory for test database
        cls.test_db_dir = tempfile.mkdtemp()
        cls.test_db_path = os.path.join(cls.test_db_dir, "test_ragdb")
        
        # Create temporary directory for test files
        cls.test_files_dir = tempfile.mkdtemp()
        
        # Create sample CSV file
        cls.sample_csv = cls._create_sample_csv()
        
        # Create sample Excel file
        cls.sample_excel = cls._create_sample_excel()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        # Remove temporary directories
        shutil.rmtree(cls.test_db_dir, ignore_errors=True)
        shutil.rmtree(cls.test_files_dir, ignore_errors=True)
    
    @classmethod
    def _create_sample_csv(cls):
        """Create a sample CSV file for testing."""
        # Create sample flight data
        data = {
            "Timestamp": pd.date_range("2024-01-15 10:00:00", periods=100, freq="1min"),
            "Altitude": np.random.uniform(1000, 10000, 100),
            "Speed": np.random.uniform(150, 300, 100),
            "Temperature": np.random.uniform(10, 25, 100),
            "Engine_Torque": np.random.uniform(70, 95, 100),
            "ITT": np.random.uniform(600, 800, 100),
            "Flight_Phase": np.random.choice(["Climb", "Cruise", "Descent"], 100)
        }
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(cls.test_files_dir, "sample_flight_data.csv")
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    @classmethod
    def _create_sample_excel(cls):
        """Create a sample Excel file for testing."""
        # Create sample data
        data = {
            "Parameter": ["Altitude", "Speed", "Temperature"],
            "Min": [0, 100, -20],
            "Max": [15000, 400, 50],
            "Unit": ["ft", "knots", "C"]
        }
        
        df = pd.DataFrame(data)
        excel_path = os.path.join(cls.test_files_dir, "sample_limits.xlsx")
        df.to_excel(excel_path, index=False)
        
        return excel_path
    
    def setUp(self):
        """Set up each test."""
        self.ingestor = DataIngestor(db_path=self.test_db_path)
    
    def test_initialization(self):
        """Test DataIngestor initialization."""
        self.assertIsNotNone(self.ingestor)
        self.assertEqual(self.ingestor.db_path, self.test_db_path)
        self.assertIsNotNone(self.ingestor.client)
        self.assertIsNotNone(self.ingestor.rows_collection)
        self.assertIsNotNone(self.ingestor.columns_collection)
        self.assertIsNotNone(self.ingestor.summaries_collection)
    
    def test_read_csv_file(self):
        """Test reading CSV file."""
        df = self.ingestor.read_file(self.sample_csv)
        
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertIn("Altitude", df.columns)
        self.assertIn("Speed", df.columns)
    
    def test_read_excel_file(self):
        """Test reading Excel file."""
        df = self.ingestor.read_file(self.sample_excel)
        
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Parameter", df.columns)
        self.assertIn("Min", df.columns)
    
    def test_create_row_text(self):
        """Test row text creation."""
        df = pd.DataFrame({
            "Altitude": [5000],
            "Speed": [250],
            "Temperature": [15.5]
        })
        
        row = df.iloc[0]
        text = self.ingestor._create_row_text(row, df.columns.tolist())
        
        self.assertIn("Altitude: 5000", text)
        self.assertIn("Speed: 250", text)
        self.assertIn("Temperature: 15.50", text)
    
    def test_create_column_text(self):
        """Test column text creation."""
        df = pd.DataFrame({
            "Altitude": [1000, 2000, 3000, 4000, 5000],
            "Speed": [150, 200, 250, 300, 350]
        })
        
        text = self.ingestor._create_column_text(df, "Altitude")
        
        self.assertIn("Column name: Altitude", text)
        self.assertIn("Data type:", text)
        self.assertIn("Range:", text)
        self.assertIn("Mean:", text)
    
    def test_create_summary_text(self):
        """Test summary text creation."""
        df = pd.DataFrame({
            "Altitude": np.random.uniform(1000, 10000, 50),
            "Speed": np.random.uniform(150, 300, 50),
            "Flight_Phase": np.random.choice(["Climb", "Cruise"], 50)
        })
        
        text = self.ingestor._create_summary_text(df, "test_data.csv")
        
        self.assertIn("test_data.csv", text)
        self.assertIn("50 rows", text)
        self.assertIn("3 columns", text)
        self.assertIn("Numeric columns", text)
    
    def test_generate_file_id(self):
        """Test file ID generation."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        file_id_1 = self.ingestor._generate_file_id("test.csv", df)
        file_id_2 = self.ingestor._generate_file_id("test.csv", df)
        
        # Same file should generate same ID
        self.assertEqual(file_id_1, file_id_2)
        
        # Different file should generate different ID
        df2 = pd.DataFrame({"A": [4, 5, 6]})
        file_id_3 = self.ingestor._generate_file_id("test.csv", df2)
        self.assertNotEqual(file_id_1, file_id_3)
    
    def test_ingest_file_csv(self):
        """Test full CSV file ingestion (without actual embedding)."""
        # Note: This test will skip actual embedding if OPENAI_API_KEY is not set
        # It mainly tests the workflow
        
        result = self.ingestor.ingest_file(
            file_path=self.sample_csv,
            file_name="sample_flight_data.csv",
            embedding_strategy="summary",  # Only summary to minimize API calls
            max_rows=10
        )
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["file_name"], "sample_flight_data.csv")
        self.assertEqual(result["row_count"], 100)
        self.assertGreater(result["column_count"], 0)
    
    def test_list_ingested_files(self):
        """Test listing ingested files."""
        # Ingest a file first
        self.ingestor.ingest_file(
            file_path=self.sample_csv,
            file_name="test_list.csv",
            embedding_strategy="summary",
            max_rows=10
        )
        
        files = self.ingestor.list_ingested_files()
        
        self.assertIsInstance(files, list)
        # Should have at least one file
        if len(files) > 0:
            self.assertIn("file_id", files[0])
            self.assertIn("file_name", files[0])
    
    def test_file_size_limit(self):
        """Test file size limit enforcement."""
        # Create a large file path (we won't actually create it)
        # We'll mock the file size check
        
        # This is a conceptual test - in practice, you'd need to create
        # a file larger than 100MB or mock os.path.getsize
        pass
    
    def test_delete_file(self):
        """Test file deletion."""
        # Ingest a file
        result = self.ingestor.ingest_file(
            file_path=self.sample_csv,
            file_name="test_delete.csv",
            embedding_strategy="summary",
            max_rows=10
        )
        
        file_id = result.get("file_id")
        
        if file_id:
            # Delete the file
            success = self.ingestor.delete_file(file_id)
            self.assertTrue(success)
            
            # Verify it's deleted
            metadata = self.ingestor.get_file_metadata(file_id)
            self.assertIsNone(metadata)


class TestEmbeddingStrategies(unittest.TestCase):
    """Test different embedding strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_db_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_db_dir, "test_ragdb")
        self.ingestor = DataIngestor(db_path=self.test_db_path)
        
        # Create sample data
        self.sample_df = pd.DataFrame({
            "Altitude": [1000, 2000, 3000, 4000, 5000],
            "Speed": [150, 200, 250, 300, 350],
            "Temperature": [10, 12, 15, 18, 20]
        })
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_db_dir, ignore_errors=True)
    
    def test_row_embedding_strategy(self):
        """Test row-level embedding generation."""
        # Note: This will only test the structure, not actual embeddings
        # unless OPENAI_API_KEY is set
        
        count = self.ingestor.generate_row_embeddings(
            df=self.sample_df,
            file_id="test_file_123",
            file_name="test.csv",
            max_rows=5
        )
        
        # Should embed all 5 rows
        self.assertEqual(count, 5)
    
    def test_column_embedding_strategy(self):
        """Test column-level embedding generation."""
        count = self.ingestor.generate_column_embeddings(
            df=self.sample_df,
            file_id="test_file_123",
            file_name="test.csv"
        )
        
        # Should embed all 3 columns
        self.assertEqual(count, 3)
    
    def test_summary_embedding_strategy(self):
        """Test summary-level embedding generation."""
        success = self.ingestor.generate_summary_embedding(
            df=self.sample_df,
            file_id="test_file_123",
            file_name="test.csv"
        )
        
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()

