"""
Demo script to test data ingestion with sample flight data.
"""

import sys
import os
sys.path.insert(0, 'enhance_flight_analyzer_rag')

import pandas as pd
from components.data_ingest import DataIngestor

def main():
    print("=" * 80)
    print("Data Ingestion Module Test")
    print("=" * 80)
    
    # Initialize ingestor
    print("\n1. Initializing DataIngestor...")
    ingestor = DataIngestor(db_path=".ragdb_test")
    print("✓ DataIngestor initialized")
    
    # Test file reading
    print("\n2. Testing file reading...")
    
    # Test Excel file
    excel_path = "sample_data.csv.xlsx"
    df_excel = ingestor.read_file(excel_path)
    
    if df_excel is not None:
        print(f"✓ Excel file read successfully")
        print(f"  Shape: {df_excel.shape}")
        print(f"  Columns: {len(df_excel.columns)}")
        
        # The first row contains units, skip it
        df_excel_clean = df_excel.iloc[1:].reset_index(drop=True)
        
        # Convert numeric columns
        for col in df_excel_clean.columns[1:]:  # Skip Description column
            try:
                df_excel_clean[col] = pd.to_numeric(df_excel_clean[col], errors='coerce')
            except:
                pass
        
        print(f"  After cleaning: {df_excel_clean.shape}")
    else:
        print("✗ Failed to read Excel file")
        return
    
    # Test CSV file
    csv_path = "Piece_Data_Sample.csv"
    df_csv = ingestor.read_file(csv_path)
    
    if df_csv is not None:
        print(f"✓ CSV file read successfully")
        print(f"  Shape: {df_csv.shape}")
        # Skip units row
        df_csv_clean = pd.read_csv(csv_path, skiprows=[1])
        print(f"  After cleaning: {df_csv_clean.shape}")
    else:
        print("✗ Failed to read CSV file")
    
    # Test text generation methods
    print("\n3. Testing text generation methods...")
    
    # Test row text
    sample_row = df_excel_clean.iloc[0]
    row_text = ingestor._create_row_text(sample_row, df_excel_clean.columns.tolist())
    print(f"✓ Row text generated ({len(row_text)} chars)")
    print(f"  Sample: {row_text[:200]}...")
    
    # Test column text
    col_text = ingestor._create_column_text(df_excel_clean, df_excel_clean.columns[1])
    print(f"✓ Column text generated ({len(col_text)} chars)")
    print(f"  Sample: {col_text[:200]}...")
    
    # Test summary text
    summary_text = ingestor._create_summary_text(df_excel_clean, "sample_data.csv.xlsx")
    print(f"✓ Summary text generated ({len(summary_text)} chars)")
    print(f"  Full text: {summary_text}")
    
    # Test file ID generation
    print("\n4. Testing file ID generation...")
    file_id = ingestor._generate_file_id("sample_data.csv.xlsx", df_excel_clean)
    print(f"✓ File ID generated: {file_id}")
    
    # Test ingestion (summary only to minimize API calls)
    print("\n5. Testing file ingestion (summary only)...")
    print("   Note: This requires OPENAI_API_KEY to be set for embeddings")
    
    # Save cleaned data to temp file
    temp_excel = "/tmp/sample_flight_data_clean.xlsx"
    df_excel_clean.to_excel(temp_excel, index=False)
    
    result = ingestor.ingest_file(
        file_path=temp_excel,
        file_name="sample_flight_data.xlsx",
        embedding_strategy="summary",  # Only summary to minimize API calls
        max_rows=10
    )
    
    print(f"\nIngestion result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Test listing files
    print("\n6. Testing file listing...")
    files = ingestor.list_ingested_files()
    print(f"✓ Found {len(files)} ingested file(s)")
    for f in files:
        print(f"  - {f.get('file_name')} ({f.get('total_rows')} rows, {f.get('total_columns')} cols)")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
