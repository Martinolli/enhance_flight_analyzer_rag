# Natural Language Querying Enhancement Plan
## Enhanced Flight Data Analyzer RAG System

---

## Executive Summary

This document outlines a comprehensive plan to enhance the Natural Language Querying capabilities of the Enhanced Flight Data Analyzer RAG system. The enhancement focuses on three core capabilities:

1. **Reading uploaded data files** directly through natural language queries
2. **Generating tables dynamically** based on query intent and data analysis
3. **Querying data using embedding vectors** for semantic search and pattern matching

The plan is structured in **5 implementation steps** with clear deliverables, timelines, and success criteria.

---

## Current State Assessment

### Strengths
- Solid RAG foundation with ChromaDB and OpenAI embeddings
- Tool-calling architecture in LLM assistant
- Streamlit-based UI with data visualization
- Basic statistical analysis capabilities
- Document knowledge base integration

### Gaps
- No embedding-based querying of uploaded CSV/Excel data
- Limited automatic table generation
- Separate workflows for KB queries and data analysis
- No unified interface for multi-source querying
- Manual tool specification required for complex queries

---

## Enhancement Goals

### Primary Objectives
1. **Enable semantic search over uploaded flight data** using embedding vectors
2. **Automate table generation** based on natural language query understanding
3. **Unify knowledge base and data querying** in a single conversational interface
4. **Improve query understanding** with intent classification and entity extraction
5. **Enhance answer quality** with multi-source citations and confidence scoring

### Success Criteria
- Users can upload CSV/Excel files and query them using natural language
- System automatically generates relevant tables without manual specification
- Queries can combine knowledge base documents and uploaded data
- Response time < 3 seconds for typical queries
- Answer accuracy > 85% (measured through user evaluation)

---

## Architecture Design

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)               │
│  - File Upload  - Query Input  - Results Display            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Query Understanding Module (NEW)                │
│  - Intent Classification  - Entity Extraction                │
│  - Execution Planning                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Enhanced LLM Assistant                        │
│  - Tool Orchestration  - Multi-round Conversation           │
│  - Answer Generation  - Citation Management                 │
└────────┬───────────────┴───────────────┬────────────────────┘
         │                               │
         ▼                               ▼
┌────────────────────┐         ┌────────────────────────────┐
│  Data Ingestion    │         │  Hybrid Retrieval          │
│  Module (NEW)      │         │  Module (ENHANCED)         │
│  - File Reading    │         │  - KB Retrieval            │
│  - Embedding Gen   │         │  - Data Retrieval          │
│  - Metadata Track  │         │  - Result Fusion           │
└────────┬───────────┘         └────────┬───────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         ▼
              ┌──────────────────────┐
              │   ChromaDB Vector    │
              │   Database           │
              │  - Documents         │
              │  - Data Embeddings   │
              └──────────────────────┘
```

---

## Implementation Steps

### Step 1: Data Ingestion Module (Week 1)

#### Objectives
- Enable reading and processing of uploaded data files
- Generate embeddings for data rows, columns, and summaries
- Store embeddings in ChromaDB with rich metadata
- Provide data preview and schema inspection

#### Components to Create

**1.1 File: `components/data_ingest.py`**

**Class: `DataIngestor`**
```python
class DataIngestor:
    """Handles ingestion of uploaded data files into the vector database."""
    
    def __init__(self, db_path: str = ".ragdb"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
    def ingest_file(
        self, 
        file_path: str, 
        file_name: str,
        embedding_strategy: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Ingest a data file and generate embeddings.
        
        Args:
            file_path: Path to the uploaded file
            file_name: Original filename
            embedding_strategy: "row", "column", "summary", or "hybrid"
            
        Returns:
            Dict with file_id, row_count, column_count, status
        """
        
    def read_file(self, file_path: str) -> pd.DataFrame:
        """Read file based on extension (csv, xlsx, parquet)."""
        
    def generate_row_embeddings(
        self, 
        df: pd.DataFrame, 
        max_rows: int = 10000
    ) -> List[Dict]:
        """Generate embeddings for data rows."""
        
    def generate_column_embeddings(
        self, 
        df: pd.DataFrame
    ) -> List[Dict]:
        """Generate embeddings for column metadata."""
        
    def generate_summary_embedding(
        self, 
        df: pd.DataFrame
    ) -> Dict:
        """Generate embedding for dataset summary."""
        
    def store_embeddings(
        self, 
        embeddings: List[Dict], 
        collection_name: str
    ) -> bool:
        """Store embeddings in ChromaDB."""
        
    def get_file_metadata(self, file_id: str) -> Dict:
        """Retrieve metadata for an ingested file."""
        
    def list_ingested_files(self) -> List[Dict]:
        """List all ingested files with metadata."""
```

**1.2 Database Schema Extension**

**New Collections**:
- `uploaded_data_rows`: Row-level embeddings
- `uploaded_data_columns`: Column metadata and embeddings
- `uploaded_data_summaries`: Dataset-level summaries

**Metadata Structure**:
```python
{
    "file_id": "uuid-string",
    "file_name": "flight_data_2024.csv",
    "file_type": "csv",
    "upload_timestamp": "2024-10-16T10:00:00Z",
    "row_index": 42,  # for row embeddings
    "column_name": "Altitude",  # for column embeddings
    "total_rows": 10000,
    "total_columns": 25,
    "embedding_type": "row|column|summary",
    "source_type": "data",
    "data_sample": "Altitude: 5000, Speed: 250, ...",
    "statistics": {
        "mean": 5000,
        "std": 1500,
        "min": 0,
        "max": 10000
    }
}
```

**1.3 Embedding Strategies**

**Row Embedding Strategy**:
```python
def create_row_embedding_text(row: pd.Series, columns: List[str]) -> str:
    """
    Create text representation of a row for embedding.
    
    Example output:
    "Timestamp: 2024-01-15 10:30:00, Altitude: 5000 ft, 
     Speed: 250 knots, Temperature: 15 C, Engine Torque: 85%"
    """
    parts = []
    for col in columns:
        value = row[col]
        if pd.notna(value):
            parts.append(f"{col}: {value}")
    return ", ".join(parts)
```

**Column Embedding Strategy**:
```python
def create_column_embedding_text(
    df: pd.DataFrame, 
    column: str
) -> str:
    """
    Create text representation of a column for embedding.
    
    Example output:
    "Column: Altitude, Type: float64, Unit: ft, 
     Range: 0 to 10000, Mean: 5000, Std: 1500,
     Description: Aircraft altitude above sea level,
     Sample values: [1000, 2000, 3000, 4000, 5000]"
    """
    stats = df[column].describe()
    samples = df[column].dropna().sample(min(5, len(df))).tolist()
    
    return f"""Column: {column}
Type: {df[column].dtype}
Range: {stats['min']} to {stats['max']}
Mean: {stats['mean']:.2f}
Std: {stats['std']:.2f}
Sample values: {samples}"""
```

**Summary Embedding Strategy**:
```python
def create_summary_embedding_text(
    df: pd.DataFrame, 
    file_name: str
) -> str:
    """
    Create text representation of dataset summary.
    
    Example output:
    "Flight data from file: flight_data_2024.csv
     Total records: 10000 rows, 25 columns
     Time range: 2024-01-15 to 2024-01-20
     Key parameters: Altitude (0-10000 ft), Speed (100-300 knots), ...
     Data quality: 98% complete, 2% missing values"
    """
```

**1.4 UI Components**

**File Upload Interface** (add to `pages/02_Report_Assistant.py`):
```python
st.subheader("📁 Upload Data for Semantic Search")
uploaded_file = st.file_uploader(
    "Upload flight data (CSV, Excel, Parquet)",
    type=["csv", "xlsx", "parquet"]
)

if uploaded_file:
    with st.spinner("Processing and embedding data..."):
        # Save file temporarily
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Ingest and embed
        ingestor = DataIngestor()
        result = ingestor.ingest_file(temp_path, uploaded_file.name)
        
        st.success(f"✅ Ingested {result['row_count']} rows, {result['column_count']} columns")
        st.info(f"File ID: {result['file_id']}")
```

#### Deliverables
- [ ] `components/data_ingest.py` with `DataIngestor` class
- [ ] Database schema migration script
- [ ] Unit tests for file reading and embedding generation
- [ ] UI component for file upload
- [ ] Documentation for embedding strategies

#### Testing
- Test CSV, Excel, Parquet file reading
- Verify embedding generation for 1K, 10K, 100K rows
- Validate metadata storage and retrieval
- Test error handling for malformed files

---

### Step 2: Hybrid Retrieval System (Week 2)

#### Objectives
- Implement unified retrieval from both documents and data
- Add metadata filtering capabilities
- Create result fusion and re-ranking logic
- Optimize retrieval performance

#### Components to Create

**2.1 File: `components/rag/hybrid_retrieval.py`**

**Class: `HybridRetriever`**
```python
class HybridRetriever:
    """Unified retrieval from knowledge base and uploaded data."""
    
    def __init__(self, db_path: str = ".ragdb"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
    def retrieve_from_kb(
        self, 
        query: str, 
        k: int = 6
    ) -> List[Dict]:
        """Retrieve from document knowledge base."""
        
    def retrieve_from_data(
        self, 
        query: str, 
        k: int = 6,
        file_id: Optional[str] = None,
        embedding_type: str = "all"
    ) -> List[Dict]:
        """
        Retrieve from uploaded data embeddings.
        
        Args:
            query: Natural language query
            k: Number of results
            file_id: Optional file filter
            embedding_type: "row", "column", "summary", or "all"
        """
        
    def retrieve_hybrid(
        self, 
        query: str, 
        k: int = 10,
        sources: List[str] = ["kb", "data"],
        weights: Dict[str, float] = {"kb": 0.5, "data": 0.5}
    ) -> List[Dict]:
        """
        Retrieve from multiple sources and fuse results.
        
        Returns results with unified format:
        {
            "text": str,
            "source_type": "kb" | "data",
            "score": float,
            "metadata": dict,
            "rank": int
        }
        """
        
    def rerank_results(
        self, 
        results: List[Dict], 
        query: str,
        strategy: str = "reciprocal_rank_fusion"
    ) -> List[Dict]:
        """Re-rank results using advanced scoring."""
        
    def filter_by_metadata(
        self, 
        results: List[Dict], 
        filters: Dict[str, Any]
    ) -> List[Dict]:
        """Apply metadata filters to results."""
```

**2.2 Retrieval Strategies**

**Reciprocal Rank Fusion (RRF)**:
```python
def reciprocal_rank_fusion(
    results_lists: List[List[Dict]], 
    k: int = 60
) -> List[Dict]:
    """
    Fuse multiple ranked lists using RRF.
    
    RRF score = sum(1 / (k + rank_i)) for each list
    """
    scores = defaultdict(float)
    for results in results_lists:
        for rank, result in enumerate(results, 1):
            result_id = result.get("id")
            scores[result_id] += 1.0 / (k + rank)
    
    # Sort by RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked
```

**Metadata Filtering**:
```python
def apply_filters(results: List[Dict], filters: Dict) -> List[Dict]:
    """
    Apply metadata filters.
    
    Example filters:
    {
        "file_name": "flight_data_2024.csv",
        "date_range": ["2024-01-01", "2024-01-31"],
        "columns": ["Altitude", "Speed"],
        "min_score": 0.7
    }
    """
```

**2.3 Enhanced Retrieval Function**

```python
def retrieve_with_context(
    query: str,
    context: Dict[str, Any],
    k: int = 10
) -> Dict[str, Any]:
    """
    Retrieve with full context awareness.
    
    Args:
        query: User's natural language query
        context: {
            "uploaded_files": List[str],  # file IDs
            "current_df": pd.DataFrame,   # active dataset
            "query_history": List[str],   # previous queries
            "user_preferences": Dict      # settings
        }
        k: Number of results
        
    Returns:
        {
            "kb_results": List[Dict],
            "data_results": List[Dict],
            "fused_results": List[Dict],
            "metadata": Dict
        }
    """
```

#### Deliverables
- [ ] `components/rag/hybrid_retrieval.py` with `HybridRetriever` class
- [ ] Retrieval benchmarking script
- [ ] Unit tests for each retrieval strategy
- [ ] Performance optimization (caching, indexing)
- [ ] Documentation for retrieval algorithms

#### Testing
- Test retrieval accuracy on sample queries
- Benchmark retrieval speed (target: < 500ms)
- Validate metadata filtering
- Test result fusion quality

---

### Step 3: Enhanced LLM Assistant Tools (Week 3)

#### Objectives
- Add new tools for data querying and table generation
- Enhance tool orchestration logic
- Improve answer generation with multi-source citations
- Add confidence scoring

#### Components to Enhance

**3.1 File: `components/llm/assistant.py` (UPDATED)**

**New Tools to Add**:

**Tool 1: `read_uploaded_file`**
```python
{
    "type": "function",
    "function": {
        "name": "read_uploaded_file",
        "description": "Read and preview an uploaded data file by file ID or name",
        "parameters": {
            "type": "object",
            "properties": {
                "file_identifier": {
                    "type": "string",
                    "description": "File ID or filename"
                },
                "preview_rows": {
                    "type": "integer",
                    "description": "Number of rows to preview (default: 10)"
                }
            },
            "required": ["file_identifier"]
        }
    }
}
```

**Tool 2: `query_data_embeddings`**
```python
{
    "type": "function",
    "function": {
        "name": "query_data_embeddings",
        "description": "Search uploaded data using natural language semantic search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing the data to find"
                },
                "file_id": {
                    "type": "string",
                    "description": "Optional file ID to search within"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)"
                },
                "embedding_type": {
                    "type": "string",
                    "enum": ["row", "column", "summary", "all"],
                    "description": "Type of embeddings to search"
                }
            },
            "required": ["query"]
        }
    }
}
```

**Tool 3: `generate_pivot_table`**
```python
{
    "type": "function",
    "function": {
        "name": "generate_pivot_table",
        "description": "Create a pivot table from uploaded data",
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "index": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to use as index"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to use as pivot columns"
                },
                "values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to aggregate"
                },
                "aggfunc": {
                    "type": "string",
                    "description": "Aggregation function (mean, sum, count, etc.)"
                }
            },
            "required": ["file_id", "index", "values"]
        }
    }
}
```

**Tool 4: `compute_aggregations`**
```python
{
    "type": "function",
    "function": {
        "name": "compute_aggregations",
        "description": "Perform groupby and aggregation operations",
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "groupby_columns": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "agg_operations": {
                    "type": "object",
                    "description": "Dict mapping column names to aggregation functions"
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters to apply before aggregation"
                }
            },
            "required": ["file_id", "groupby_columns", "agg_operations"]
        }
    }
}
```

**Tool 5: `find_similar_records`**
```python
{
    "type": "function",
    "function": {
        "name": "find_similar_records",
        "description": "Find similar rows in uploaded data using embeddings",
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "reference_row_index": {
                    "type": "integer",
                    "description": "Index of the reference row"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of similar records to find"
                },
                "columns_to_compare": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: specific columns to use for similarity"
                }
            },
            "required": ["file_id", "reference_row_index"]
        }
    }
}
```

**Tool 6: `filter_data`**
```python
{
    "type": "function",
    "function": {
        "name": "filter_data",
        "description": "Apply complex filters to uploaded data",
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "operator": {
                                "type": "string",
                                "enum": [">", "<", ">=", "<=", "==", "!=", "in", "between"]
                            },
                            "value": {}
                        }
                    }
                },
                "logic": {
                    "type": "string",
                    "enum": ["AND", "OR"],
                    "description": "How to combine conditions"
                }
            },
            "required": ["file_id", "conditions"]
        }
    }
}
```

**3.2 Tool Handler Implementation**

```python
def _handle_tool_call(
    self,
    name: str,
    arguments: Dict[str, Any],
    df: Optional[pd.DataFrame],
    context: Dict[str, Any]
) -> str:
    """Enhanced tool handler with context awareness."""
    
    if name == "read_uploaded_file":
        return self._handle_read_file(arguments, context)
    
    elif name == "query_data_embeddings":
        return self._handle_query_embeddings(arguments, context)
    
    elif name == "generate_pivot_table":
        return self._handle_pivot_table(arguments, context)
    
    elif name == "compute_aggregations":
        return self._handle_aggregations(arguments, context)
    
    elif name == "find_similar_records":
        return self._handle_find_similar(arguments, context)
    
    elif name == "filter_data":
        return self._handle_filter_data(arguments, context)
    
    # ... existing tools ...
```

**3.3 Enhanced Answer Generation**

```python
def ask(
    self,
    prompt: str,
    system_prompt: str,
    df: Optional[pd.DataFrame] = None,
    context: Optional[Dict] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    detail_level: str = "standard",
    enable_tools: bool = True,
    enable_citations: bool = True
) -> Dict[str, Any]:
    """
    Enhanced ask method with multi-source support.
    
    Returns:
    {
        "text": str,                          # Main answer
        "sections": Dict[str, str],           # Structured sections
        "tables": List[Tuple[str, DataFrame]], # Generated tables
        "suggestions": List[Dict],            # Calculation suggestions
        "citations": List[Dict],              # Multi-source citations
        "confidence": float,                  # Answer confidence score
        "sources_used": List[str]             # KB, data, model knowledge
    }
    """
```

#### Deliverables
- [ ] Updated `components/llm/assistant.py` with new tools
- [ ] Tool handler implementations
- [ ] Enhanced answer generation logic
- [ ] Citation and confidence scoring
- [ ] Unit tests for each tool
- [ ] Integration tests for tool orchestration

#### Testing
- Test each tool individually
- Verify tool chaining (multi-tool queries)
- Validate table generation quality
- Test citation accuracy

---

### Step 4: Query Understanding Module (Week 4)

#### Objectives
- Implement query intent classification
- Extract entities (columns, metrics, filters)
- Generate execution plans
- Optimize query routing

#### Components to Create

**4.1 File: `components/query_understanding.py`**

**Class: `QueryAnalyzer`**
```python
class QueryAnalyzer:
    """Analyzes natural language queries to determine intent and execution plan."""
    
    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
        
    def classify_intent(self, query: str) -> QueryIntent:
        """
        Classify query into one of:
        - KB_SEARCH: Knowledge base lookup
        - DATA_ANALYSIS: Data computation/analysis
        - HYBRID: Requires both KB and data
        - GENERAL: General question (model knowledge)
        """
        
    def extract_entities(
        self, 
        query: str, 
        schema: Dict[str, Any]
    ) -> List[Entity]:
        """
        Extract entities from query.
        
        Entity types:
        - COLUMN: Column names mentioned
        - METRIC: Statistical measures (mean, max, etc.)
        - FILTER: Conditions (altitude > 5000)
        - TIME_RANGE: Temporal filters
        - AGGREGATION: Grouping operations
        """
        
    def plan_execution(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Generate execution plan.
        
        Returns:
        {
            "intent": QueryIntent,
            "entities": List[Entity],
            "required_tools": List[str],
            "data_sources": List[str],
            "execution_order": List[str],
            "estimated_complexity": str  # low, medium, high
        }
        """
```

**4.2 Intent Classification**

```python
def classify_intent_with_llm(query: str, context: Dict) -> str:
    """
    Use LLM to classify query intent.
    
    System prompt:
    "You are a query classifier for a flight data analysis system.
    Classify the user's query into one of:
    - KB_SEARCH: Needs information from knowledge base documents
    - DATA_ANALYSIS: Needs computation on uploaded data
    - HYBRID: Needs both knowledge base and data analysis
    - GENERAL: General question answerable with model knowledge
    
    Consider:
    - Available uploaded files: {context['files']}
    - Available KB topics: {context['kb_topics']}
    
    Query: {query}
    
    Respond with JSON:
    {
        'intent': 'KB_SEARCH|DATA_ANALYSIS|HYBRID|GENERAL',
        'confidence': 0.0-1.0,
        'reasoning': 'brief explanation'
    }"
    """
```

**4.3 Entity Extraction**

```python
def extract_entities_with_llm(
    query: str, 
    schema: Dict
) -> List[Dict]:
    """
    Extract entities using LLM with schema awareness.
    
    System prompt:
    "Extract entities from the query given the data schema.
    
    Available columns: {schema['columns']}
    
    Extract:
    - Column references (exact or fuzzy matches)
    - Metrics (mean, max, min, count, etc.)
    - Filters (conditions like > 5000, between X and Y)
    - Time ranges
    - Aggregation operations
    
    Query: {query}
    
    Respond with JSON array of entities:
    [
        {
            'type': 'COLUMN',
            'value': 'Altitude',
            'original_text': 'altitude',
            'confidence': 0.95
        },
        ...
    ]"
    """
```

**4.4 Execution Planning**

```python
def create_execution_plan(
    intent: str,
    entities: List[Dict],
    context: Dict
) -> Dict:
    """
    Create execution plan based on intent and entities.
    
    Example plan for HYBRID query:
    {
        "steps": [
            {
                "step": 1,
                "action": "retrieve_from_kb",
                "params": {"query": "...", "k": 5}
            },
            {
                "step": 2,
                "action": "query_data_embeddings",
                "params": {"query": "...", "file_id": "..."}
            },
            {
                "step": 3,
                "action": "compute_stats",
                "params": {"columns": ["Altitude", "Speed"]}
            },
            {
                "step": 4,
                "action": "generate_answer",
                "params": {"sources": ["kb", "data", "stats"]}
            }
        ],
        "estimated_time": "2-3 seconds",
        "complexity": "medium"
    }
    """
```

#### Deliverables
- [ ] `components/query_understanding.py` with `QueryAnalyzer` class
- [ ] Intent classification logic
- [ ] Entity extraction with fuzzy matching
- [ ] Execution planning algorithm
- [ ] Unit tests for each component
- [ ] Benchmark for classification accuracy

#### Testing
- Test intent classification on 100+ sample queries
- Validate entity extraction accuracy
- Test execution plan generation
- Measure classification speed (target: < 200ms)

---

### Step 5: UI Integration and Testing (Week 5)

#### Objectives
- Integrate all components into unified interface
- Create comprehensive testing suite
- Optimize performance
- Document usage and examples

#### UI Enhancements

**5.1 Enhanced Report Assistant Page**

```python
# pages/02_Report_Assistant.py (ENHANCED)

st.title("🧭 Enhanced Knowledge & Data Assistant")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs([
    "💬 Ask Questions", 
    "📁 Manage Data", 
    "📊 Analytics"
])

with tab1:
    st.subheader("Natural Language Query Interface")
    
    # Query input
    query = st.text_area(
        "Ask a question about your data or knowledge base",
        placeholder="e.g., What was the average altitude during the high-speed flight test? Compare with recommended limits from the manual.",
        height=100
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            sources = st.multiselect(
                "Data sources",
                ["Knowledge Base", "Uploaded Data", "Model Knowledge"],
                default=["Knowledge Base", "Uploaded Data"]
            )
            detail_level = st.select_slider(
                "Detail level",
                options=["Brief", "Standard", "Deep"],
                value="Standard"
            )
        with col2:
            enable_tools = st.checkbox("Enable smart tools", value=True)
            enable_citations = st.checkbox("Show citations", value=True)
            top_k = st.slider("Results per source", 1, 20, 5)
    
    if st.button("🔍 Search & Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question")
        else:
            # Query understanding
            with st.spinner("Understanding query..."):
                analyzer = QueryAnalyzer(llm_client)
                plan = analyzer.plan_execution(query, context)
                
                st.info(f"Query type: {plan['intent']} | Complexity: {plan['estimated_complexity']}")
            
            # Execute query
            with st.spinner("Retrieving and analyzing..."):
                assistant = ToolEnabledLLM(api_key=OPENAI_API_KEY)
                result = assistant.ask(
                    prompt=query,
                    system_prompt=build_system_prompt(plan),
                    context=build_context(plan),
                    detail_level=detail_level.lower(),
                    enable_tools=enable_tools
                )
            
            # Display results
            st.markdown("### Answer")
            st.markdown(result["text"])
            
            # Confidence score
            confidence = result.get("confidence", 0.0)
            st.metric("Confidence", f"{confidence:.1%}")
            
            # Citations
            if enable_citations and result.get("citations"):
                with st.expander("📚 Citations & Sources"):
                    for i, citation in enumerate(result["citations"], 1):
                        st.markdown(f"**[{i}] {citation['source_type'].upper()}**")
                        st.caption(citation.get("metadata", {}).get("source", ""))
                        st.text(citation["text"][:300] + "...")
            
            # Generated tables
            if result.get("tables"):
                st.markdown("### Generated Tables")
                for table_name, table_df in result["tables"]:
                    with st.expander(f"📊 {table_name}"):
                        st.dataframe(table_df, use_container_width=True)
                        csv = table_df.to_csv(index=False).encode()
                        st.download_button(
                            f"Download {table_name}",
                            data=csv,
                            file_name=f"{table_name.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
            
            # Suggestions
            if result.get("suggestions"):
                st.markdown("### 💡 Suggested Follow-up Analyses")
                for i, suggestion in enumerate(result["suggestions"], 1):
                    st.markdown(f"{i}. **{suggestion['title']}** — {suggestion['description']}")

with tab2:
    st.subheader("📁 Data Management")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload flight data",
        type=["csv", "xlsx", "parquet"],
        help="Upload CSV, Excel, or Parquet files for semantic search"
    )
    
    if uploaded_file:
        # Upload and ingest
        with st.spinner("Processing and embedding..."):
            ingestor = DataIngestor()
            result = ingestor.ingest_file(uploaded_file)
            st.success(f"✅ Processed {result['row_count']} rows")
    
    # List ingested files
    st.markdown("#### Ingested Files")
    ingestor = DataIngestor()
    files = ingestor.list_ingested_files()
    
    if files:
        files_df = pd.DataFrame(files)
        st.dataframe(files_df, use_container_width=True)
    else:
        st.info("No files ingested yet")

with tab3:
    st.subheader("📊 Analytics Dashboard")
    
    # Quick stats
    if st.session_state.get("data") is not None:
        df = st.session_state["data"]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage().sum() / 1024**2:.1f} MB")
        col4.metric("Completeness", f"{(1 - df.isnull().sum().sum() / df.size) * 100:.1f}%")
        
        # Column explorer
        st.markdown("#### Column Explorer")
        selected_col = st.selectbox("Select column", df.columns)
        
        if selected_col:
            col_stats = df[selected_col].describe()
            st.write(col_stats)
            
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                st.plotly_chart(
                    px.histogram(df, x=selected_col),
                    use_container_width=True
                )
```

**5.2 Testing Suite**

**File: `tests/test_nl_query_enhancements.py`**

```python
import pytest
from components.data_ingest import DataIngestor
from components.rag.hybrid_retrieval import HybridRetriever
from components.llm.assistant import ToolEnabledLLM
from components.query_understanding import QueryAnalyzer

class TestDataIngestion:
    def test_csv_ingestion(self):
        """Test CSV file ingestion and embedding."""
        
    def test_excel_ingestion(self):
        """Test Excel file ingestion."""
        
    def test_row_embedding_generation(self):
        """Test row-level embedding generation."""
        
    def test_column_embedding_generation(self):
        """Test column-level embedding generation."""
        
    def test_metadata_storage(self):
        """Test metadata storage and retrieval."""

class TestHybridRetrieval:
    def test_kb_retrieval(self):
        """Test knowledge base retrieval."""
        
    def test_data_retrieval(self):
        """Test data embedding retrieval."""
        
    def test_hybrid_retrieval(self):
        """Test hybrid retrieval and fusion."""
        
    def test_metadata_filtering(self):
        """Test metadata-based filtering."""
        
    def test_reranking(self):
        """Test result re-ranking."""

class TestLLMTools:
    def test_read_uploaded_file(self):
        """Test read_uploaded_file tool."""
        
    def test_query_data_embeddings(self):
        """Test query_data_embeddings tool."""
        
    def test_generate_pivot_table(self):
        """Test pivot table generation."""
        
    def test_compute_aggregations(self):
        """Test aggregation operations."""
        
    def test_find_similar_records(self):
        """Test similarity search."""

class TestQueryUnderstanding:
    def test_intent_classification(self):
        """Test query intent classification."""
        
    def test_entity_extraction(self):
        """Test entity extraction."""
        
    def test_execution_planning(self):
        """Test execution plan generation."""

class TestEndToEnd:
    def test_simple_data_query(self):
        """Test: 'What is the average altitude?'"""
        
    def test_hybrid_query(self):
        """Test: 'Compare actual altitude with recommended limits from manual'"""
        
    def test_table_generation(self):
        """Test: 'Show me a table of altitude statistics by flight phase'"""
        
    def test_similarity_search(self):
        """Test: 'Find flights similar to the one on 2024-01-15'"""
```

#### Performance Optimization

**Caching Strategy**:
```python
# Cache embeddings for frequently accessed data
@st.cache_data(ttl=3600)
def get_cached_embeddings(file_id: str):
    """Cache embeddings for 1 hour."""
    
# Cache retrieval results
@st.cache_data(ttl=600)
def get_cached_retrieval(query: str, sources: List[str]):
    """Cache retrieval results for 10 minutes."""
```

**Batch Processing**:
```python
# Batch embed rows for efficiency
def embed_rows_batch(rows: List[str], batch_size: int = 100):
    """Embed rows in batches to optimize API calls."""
    embeddings = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        batch_embeddings = embed_texts(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

#### Deliverables
- [ ] Enhanced UI with tabs and advanced options
- [ ] Comprehensive test suite (unit + integration)
- [ ] Performance optimization (caching, batching)
- [ ] User documentation with examples
- [ ] Developer documentation
- [ ] Performance benchmarks

#### Testing
- Run full test suite (target: 100% pass rate)
- Performance testing (target: < 3s response time)
- User acceptance testing with sample queries
- Load testing (concurrent users)

---

## Timeline and Milestones

| Week | Phase | Deliverables | Success Criteria |
|------|-------|--------------|------------------|
| 1 | Data Ingestion | `data_ingest.py`, DB schema, UI upload | CSV/Excel ingestion working |
| 2 | Hybrid Retrieval | `hybrid_retrieval.py`, fusion logic | Retrieval < 500ms, accuracy > 80% |
| 3 | Enhanced Tools | New LLM tools, handlers | All 6 tools functional |
| 4 | Query Understanding | `query_understanding.py`, planning | Intent classification > 85% |
| 5 | Integration & Testing | Enhanced UI, tests, docs | All tests pass, < 3s response |

---

## Risk Mitigation

### Technical Risks

**Risk 1: Embedding generation too slow for large files**
- **Mitigation**: Implement batch processing, async embedding, row sampling
- **Fallback**: Limit to first 10K rows, provide warning to user

**Risk 2: Retrieval quality degradation with multiple sources**
- **Mitigation**: Implement re-ranking, tune fusion weights, add relevance feedback
- **Fallback**: Allow user to select specific sources

**Risk 3: Tool orchestration complexity**
- **Mitigation**: Implement execution planning, add timeout limits, provide fallbacks
- **Fallback**: Simplify to single-tool queries if multi-tool fails

### Operational Risks

**Risk 1: Increased API costs from embeddings**
- **Mitigation**: Use `text-embedding-3-small`, implement caching, batch requests
- **Monitoring**: Track API usage, set budget alerts

**Risk 2: Database size growth**
- **Mitigation**: Implement data retention policies, compression, archival
- **Monitoring**: Track DB size, alert at thresholds

---

## Success Metrics

### Functional Metrics
- ✅ Support CSV, Excel, Parquet uploads
- ✅ Generate embeddings for uploaded data
- ✅ Query data using natural language
- ✅ Automatic table generation
- ✅ Multi-source querying (KB + data)
- ✅ Accurate citations

### Performance Metrics
- ⏱️ Embedding generation: < 5s for 10K rows
- ⏱️ Query response time: < 3s (p95)
- ⏱️ Retrieval latency: < 500ms
- 💾 Support files up to 100MB
- 👥 Handle 10+ concurrent users

### Quality Metrics
- 🎯 Answer accuracy: > 85% (human eval)
- 📚 Citation accuracy: > 90%
- 📊 Table relevance: > 80%
- 😊 User satisfaction: > 4/5

---

## Next Steps

### Immediate Actions
1. **Review this plan** with stakeholders
2. **Set up development branch** in Git
3. **Create project board** with tasks
4. **Begin Step 1** (Data Ingestion Module)

### Questions for Clarification
1. **Embedding model**: Stick with `text-embedding-3-small` or upgrade?
2. **File size limits**: What's the maximum file size to support?
3. **Retention policy**: How long to keep uploaded data embeddings?
4. **Privacy**: Any special handling for sensitive flight data?
5. **Deployment**: On-premise or cloud deployment?

---

## Appendix

### Example Queries

**Query 1: Simple data analysis**
```
"What was the average engine torque during the flight?"
```
**Expected behavior**:
- Intent: DATA_ANALYSIS
- Tools: compute_stats
- Result: Table with torque statistics

**Query 2: Hybrid query**
```
"Compare the actual ITT values with the recommended limits from the aircraft manual."
```
**Expected behavior**:
- Intent: HYBRID
- Tools: query_data_embeddings, retrieve_from_kb, create_table
- Result: Comparison table with citations

**Query 3: Similarity search**
```
"Find other flights with similar altitude and speed profiles to the one on 2024-01-15."
```
**Expected behavior**:
- Intent: DATA_ANALYSIS
- Tools: find_similar_records, create_table
- Result: List of similar flights with similarity scores

**Query 4: Complex aggregation**
```
"Show me a pivot table of average torque by flight phase and altitude band."
```
**Expected behavior**:
- Intent: DATA_ANALYSIS
- Tools: generate_pivot_table
- Result: Pivot table with torque averages

---

### References
- ChromaDB documentation: https://docs.trychroma.com/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- Streamlit documentation: https://docs.streamlit.io/
- Reciprocal Rank Fusion: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

---

**Document Version**: 1.0  
**Last Updated**: October 16, 2025  
**Author**: Manus AI Assistant  
**Status**: Ready for Review

