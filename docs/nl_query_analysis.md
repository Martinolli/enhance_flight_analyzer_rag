# Natural Language Querying Enhancement Analysis

## Current System Architecture

### Overview

The Enhanced Flight Data Analyzer is a Streamlit-based application for analyzing flight test data with RAG (Retrieval-Augmented Generation) capabilities. The system currently supports:

1. **Data Upload & Visualization**: CSV/Excel flight data processing with Plotly charts
2. **Knowledge Base Q&A**: RAG-based question answering using ChromaDB vector database
3. **Report Generation**: Automated flight test report creation
4. **Statistical Analysis**: Basic statistics and frequency analysis

### Current NL Querying Components

#### 1. **LLM Assistant** (`components/llm/assistant.py`)

**Class**: `ToolEnabledLLM`

**Current Capabilities**:

- Tool-calling architecture using OpenAI function calling
- Data-aware operations on uploaded DataFrames
- Multi-round conversation with tool execution

**Existing Tools**:

1. `peek_columns`: List available columns and data types

2. `compute_stats`: Calculate statistics (count, mean, std, min, max, median, p10, p90)
3. `create_table`: Generate structured tabular outputs
4. `suggest_calculations`: Propose actionable calculations

**Limitations**:

- No direct data reading from uploaded files (only DataFrame in session state)
- No embedding vector-based querying of uploaded data
- Limited table generation (manual specification required)
- No automatic insight generation
- No integration between RAG knowledge base and uploaded data analysis

#### 2. **RAG System** (`components/rag/`)

**Components**:

- `bootstrap.py`: Database initialization
- `ingest.py`: Document embedding and storage
- `retrieval.py`: Vector similarity search

**Current Flow**:

- Documents stored in `.ragdb` using ChromaDB
- Uses OpenAI embeddings (`text-embedding-3-small`)
- Retrieves top-k passages based on query similarity
- No integration with uploaded flight data

**Limitations**:

- Only retrieves from pre-ingested documents
- No capability to embed and query uploaded CSV/Excel data
- No hybrid search (vector + metadata filtering)
- No re-ranking or relevance scoring

#### 3. **Report Assistant Page** (`pages/02_Report_Assistant.py`)

**Current Features**:

- Knowledge base Q&A with citation
- Dataset summary generation
- Report generation from templates

**Limitations**:

- Separate workflows for KB queries and data analysis
- No unified interface for querying both KB and uploaded data
- Limited dynamic table generation
- No embedding-based data querying

---

## Enhancement Requirements

### Phase 1: Enhanced Natural Language Querying

Based on user requirements, the following capabilities need to be added:

#### 1. **Read Uploaded Data**

- Direct file reading from user uploads (CSV, Excel, Parquet)
- Automatic schema detection and metadata extraction
- Support for multiple file uploads and session management
- Data preview and validation

#### 2. **Generate Tables in Reports**

- Automatic table generation based on query intent
- Smart column selection and aggregation
- Support for pivot tables, groupby operations
- Export capabilities (CSV, Excel, Markdown)

#### 3. **Read Data Using Embedding Vectors**

- Embed uploaded data rows/columns for semantic search
- Hybrid search: combine metadata filters with vector similarity
- Query data using natural language descriptions
- Find similar records or patterns in uploaded data

#### 4. **Respond to Questions Based on Embeddings**

- Answer questions about uploaded data using RAG approach
- Combine knowledge base context with data insights
- Generate answers with citations from both KB and data
- Support complex analytical queries

---

## Proposed Enhancement Plan

### Architecture Changes

#### 1. **Data Ingestion Module** (NEW)

**File**: `components/data_ingest.py`

**Responsibilities**:

- Read multiple file formats (CSV, Excel, Parquet, JSON)
- Extract metadata (column names, types, statistics)
- Generate embeddings for data rows/columns
- Store in vector database with metadata

**Key Functions**:

```python
class DataIngestor:
    def ingest_file(file_path: str, metadata: dict) -> str
    def embed_dataframe(df: pd.DataFrame, strategy: str) -> List[Embedding]
    def store_embeddings(embeddings: List, metadata: dict) -> bool
    def get_data_schema(file_id: str) -> dict
```

**Embedding Strategies**:

- **Row-level**: Embed each row as a concatenated string
- **Column-level**: Embed column descriptions and sample values
- **Chunk-level**: Embed sliding windows of data for time-series
- **Summary-level**: Embed statistical summaries

#### 2. **Hybrid Retrieval Module** (ENHANCED)

**File**: `components/rag/hybrid_retrieval.py`

**Enhancements**:

- Unified retrieval from both documents and data
- Metadata filtering (file type, date range, columns)
- Re-ranking based on relevance scores
- Fusion of multiple retrieval strategies

**Key Functions**:

```python
class HybridRetriever:
    def retrieve_from_kb(query: str, k: int) -> List[Document]
    def retrieve_from_data(query: str, k: int, filters: dict) -> List[DataChunk]
    def retrieve_hybrid(query: str, k: int, sources: List[str]) -> List[Result]
    def rerank_results(results: List, query: str) -> List[Result]
```

#### 3. **Enhanced LLM Assistant** (UPDATED)

**File**: `components/llm/assistant.py`

**New Tools**:

1. `read_uploaded_file`: Read and preview uploaded data files
2. `query_data_embeddings`: Search uploaded data using embeddings
3. `generate_pivot_table`: Create pivot tables from data
4. `compute_aggregations`: Perform groupby and aggregation operations
5. `find_similar_records`: Find similar rows using embeddings
6. `filter_data`: Apply complex filters to data
7. `join_datasets`: Merge multiple uploaded datasets

**Enhanced Capabilities**:

- Multi-source querying (KB + uploaded data)
- Automatic table generation based on query intent
- Citation from both documents and data sources
- Confidence scoring for answers

#### 4. **Query Understanding Module** (NEW)

**File**: `components/query_understanding.py`

**Responsibilities**:

- Classify query intent (KB search, data analysis, hybrid)
- Extract entities (column names, metrics, filters)
- Determine required tools and data sources
- Generate execution plan

**Key Functions**:

```python
class QueryAnalyzer:
    def classify_intent(query: str) -> QueryIntent
    def extract_entities(query: str, schema: dict) -> List[Entity]
    def plan_execution(query: str, context: dict) -> ExecutionPlan
```

---

## Implementation Roadmap

### Step 1: Data Ingestion Enhancement

**Tasks**:

1. Create `DataIngestor` class with multi-format support
2. Implement embedding strategies for data
3. Extend ChromaDB schema to support data embeddings
4. Add metadata tracking for uploaded files
5. Create UI for file upload and preview

**Deliverables**:

- `components/data_ingest.py`
- Updated database schema
- File upload interface in Streamlit

### Step 2: Hybrid Retrieval System

**Tasks**:

1. Create `HybridRetriever` class
2. Implement metadata filtering
3. Add re-ranking logic
4. Create unified result format
5. Test retrieval quality

**Deliverables**:

- `components/rag/hybrid_retrieval.py`
- Unit tests for retrieval
- Benchmark results

### Step 3: Enhanced Tool Set for LLM

**Tasks**:

1. Add new tools to `ToolEnabledLLM`
2. Implement data querying tools
3. Add table generation tools
4. Create aggregation and filtering tools
5. Test tool execution

**Deliverables**:

- Updated `components/llm/assistant.py`
- Tool documentation
- Integration tests

### Step 4: Query Understanding

**Tasks**:

1. Create `QueryAnalyzer` class
2. Implement intent classification
3. Add entity extraction
4. Create execution planning logic
5. Integrate with LLM assistant

**Deliverables**:

- `components/query_understanding.py`
- Intent classification tests
- Query examples and benchmarks

### Step 5: UI Integration

**Tasks**:

1. Update Report Assistant page
2. Add file upload and management UI
3. Create result visualization components
4. Add citation and source tracking
5. Implement export functionality

**Deliverables**:

- Updated `pages/02_Report_Assistant.py`
- UI components for data management
- Export templates

---

## Technical Specifications

### Database Schema Extension

**Current Collections**:

- `flight_test_kb`: Document embeddings

**New Collections**:

- `uploaded_data_rows`: Row-level embeddings
- `uploaded_data_columns`: Column metadata and embeddings
- `uploaded_data_summaries`: Statistical summary embeddings

**Metadata Fields**:

```json
{
  "file_id": "uuid",
  "file_name": "string",
  "file_type": "csv|excel|parquet",
  "upload_timestamp": "datetime",
  "row_index": "int",
  "columns": ["list"],
  "data_type": "row|column|summary",
  "source_type": "document|data"
}
```

### Embedding Strategy

**For Uploaded Data**:

1. **Row Embeddings**: Concatenate column values with column names
   - Example: "Altitude: 5000, Speed: 250, Temperature: 15"
   - Use for finding similar flight conditions

2. **Column Embeddings**: Embed column name + statistics + sample values
   - Example: "Altitude (ft): min=0, max=10000, mean=5000, samples=[1000, 2000, 3000]"
   - Use for schema understanding

3. **Summary Embeddings**: Embed statistical summaries and insights
   - Example: "Flight data from 2024-01-15, 10000 rows, altitude range 0-10000 ft, average speed 250 knots"
   - Use for high-level queries

### Tool Specifications

**New Tool: `read_uploaded_file`**

```json
{
  "name": "read_uploaded_file",
  "description": "Read and preview an uploaded data file",
  "parameters": {
    "file_id": "string",
    "preview_rows": "int (default: 10)"
  },
  "returns": {
    "schema": "dict",
    "preview": "DataFrame",
    "statistics": "dict"
  }
}
```

**New Tool: `query_data_embeddings`**

```json
{
  "name": "query_data_embeddings",
  "description": "Search uploaded data using natural language",
  "parameters": {
    "query": "string",
    "file_id": "string (optional)",
    "k": "int (default: 5)",
    "filters": "dict (optional)"
  },
  "returns": {
    "results": "List[DataChunk]",
    "scores": "List[float]"
  }
}
```

**New Tool: `generate_pivot_table`**

```json
{
  "name": "generate_pivot_table",
  "description": "Create a pivot table from uploaded data",
  "parameters": {
    "file_id": "string",
    "index": "string or List[string]",
    "columns": "string or List[string]",
    "values": "string or List[string]",
    "aggfunc": "string (default: mean)"
  },
  "returns": {
    "table": "DataFrame",
    "name": "string"
  }
}
```

---

## Testing Strategy

### Unit Tests

1. Data ingestion for each file format

2. Embedding generation and storage
3. Retrieval accuracy for different query types
4. Tool execution correctness
5. Query intent classification

### Integration Tests

1. End-to-end query flow (upload → embed → query → answer)
2. Hybrid retrieval (KB + data)
3. Multi-file querying
4. Citation accuracy

### Performance Tests

1. Embedding generation speed
2. Retrieval latency
3. Memory usage for large datasets
4. Concurrent user handling

---

## Success Metrics

### Functionality

- [ ] Support CSV, Excel, Parquet file uploads
- [ ] Generate embeddings for uploaded data
- [ ] Query data using natural language
- [ ] Generate tables automatically based on queries
- [ ] Combine KB and data sources in answers
- [ ] Provide accurate citations

### Performance

- [ ] Embedding generation < 5s for 10K rows
- [ ] Query response time < 3s
- [ ] Support files up to 100MB
- [ ] Handle 10+ concurrent uploads

### Quality

- [ ] Answer accuracy > 85% (human evaluation)
- [ ] Citation accuracy > 90%
- [ ] Table generation relevance > 80%
- [ ] User satisfaction score > 4/5

---

## Next Steps

1. **Review and Approve Plan**: Confirm architecture and approach
2. **Set Up Development Environment**: Install dependencies, create branches
3. **Implement Step 1**: Data ingestion module
4. **Test and Iterate**: Validate each component before proceeding
5. **Integrate and Deploy**: Combine all components and test end-to-end

---

## Open Questions

1. **Embedding Model**: Continue with `text-embedding-3-small` or upgrade to `text-embedding-3-large`?
2. **Database Scaling**: Should we consider a separate database for data embeddings?
3. **Privacy**: How to handle sensitive flight data in embeddings?
4. **Caching**: Should we cache embeddings for frequently queried data?
5. **Multi-user**: How to handle concurrent uploads and queries?

---

## References

- Current codebase: `enhance_flight_analyzer_rag`
- RAG documentation: `docs/RAG_SETUP.md`
- LLM assistant: `components/llm/assistant.py`
- Retrieval system: `components/rag/retrieval.py`
