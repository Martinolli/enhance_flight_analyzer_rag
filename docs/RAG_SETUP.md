# RAG (Retrieval-Augmented Generation) Setup Guide

## Overview

The Enhanced Flight Data Analyzer includes a powerful RAG (Retrieval-Augmented Generation) system that allows you to:

- Upload flight test knowledge documents (PDFs, text files)
- Ask natural language questions about flight testing concepts
- Generate intelligent reports based on your flight data and knowledge base
- Get contextual answers from technical documentation

This guide provides detailed instructions for setting up and using the RAG system.

## Prerequisites

### Required Dependencies

Ensure these packages are installed in your environment (check `requirements.txt`):

```bash
# Core RAG dependencies
chromadb>=0.4.0          # Vector database for document storage
pypdf>=3.0.0             # PDF text extraction
openai>=1.0.0            # OpenAI API for embeddings and chat completion
sentence-transformers    # Local embedding models (fallback)

# Optional but recommended
tiktoken                 # Token counting for optimization
python-dotenv           # Environment variable management
```

### Environment Setup

Create a `.env` file in your project root:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Embedding Model Selection (recommended)
OPENAI_EMBED_MODEL=text-embedding-3-small  # Cost-effective, 1536 dimensions
# Alternative options:
# OPENAI_EMBED_MODEL=text-embedding-3-large  # Higher quality, 3072 dimensions (more expensive)
# OPENAI_EMBED_MODEL=text-embedding-ada-002  # Legacy model, 1536 dimensions
```

**Important**: The system uses `text-embedding-3-small` by default for optimal cost-performance balance.

## Document Preparation

### 1. Create Knowledge Base Directory

Organize your flight test documents in the knowledge base folder:

```bash
docs/knowledge_base/
├── Flight_Test_Handbook.pdf
├── Aircraft_Systems_Manual.pdf
├── Test_Procedures_Guide.pdf
├── Certification_Requirements.pdf
├── vibration_analysis_notes.txt
├── control_systems_theory.md
└── safety_protocols.txt
```

### 2. Supported File Formats

- **PDF Files** (`.pdf`): Technical manuals, handbooks, reports
- **Text Files** (`.txt`): Notes, procedures, plain text documentation
- **Markdown Files** (`.md`): Formatted documentation, README files

### 3. Document Guidelines

**For Best Results:**

- Use high-quality, text-based PDFs (not scanned images)
- Ensure documents are relevant to flight testing
- Include diverse content: procedures, theory, troubleshooting, standards
- Maintain reasonable file sizes (< 50MB per document)
- Use descriptive filenames

**Content Recommendations:**

- Flight test procedures and methodologies
- Aircraft systems documentation
- Aerodynamic theory and principles
- Instrumentation and data acquisition guides
- Safety protocols and emergency procedures
- Regulatory requirements and standards
- Case studies and lessons learned

## Database Creation and Management

### Initial Database Creation

- **Method 1: Using the Convenience Script (Recommended)**

```bash
# After deleting any existing .ragdb folder
python reingest_documents.py
```

This script will:

- Check for existing database conflicts
- Verify document directory exists
- Display current embedding dimensions
- Process all documents with progress indication
- Verify successful ingestion

- **Method 2: Manual Ingestion**

```python
from components.rag.ingest import ingest_documents

# Ingest documents with custom parameters
ingest_documents(
    source_dir="docs/knowledge_base",
    db_path=".ragdb",
    chunk_size=500  # Adjust based on document complexity
)
```

### Understanding the Ingestion Process

**What Happens During Ingestion:**

1. **Document Discovery**: Scans the knowledge base directory recursively
2. **Text Extraction**:
   - PDFs: Uses pypdf to extract text content
   - Text files: Direct UTF-8 encoding reading
3. **Text Chunking**: Splits documents into manageable chunks (default: 500 characters)
4. **Embedding Generation**:
   - Creates 1536-dimensional vectors using OpenAI text-embedding-3-small
   - Processes in batches of 100 for efficiency
   - Falls back to sentence-transformers if OpenAI unavailable
5. **Vector Storage**: Stores embeddings and metadata in ChromaDB
6. **Indexing**: Creates searchable index for fast retrieval

**Typical Processing Times:**

- Small document (< 10 pages): 10-30 seconds
- Medium document (10-100 pages): 1-5 minutes
- Large document (100+ pages): 5-15 minutes

### Database Management

**Check Database Status:**

```python
import chromadb
from pathlib import Path

if Path(".ragdb").exists():
    client = chromadb.PersistentClient(path=".ragdb")
    coll = client.get_collection("flight_test_kb")
    print(f"Documents in database: {coll.count()}")
else:
    print("No database found")
```

**Database Location:**

- Default path: `.ragdb/` in project root
- Contains ChromaDB SQLite files and metadata
- Size: Typically 10-100MB depending on document volume

## Embedding Configuration

### Current Setup (Optimized)

The system is configured to use OpenAI embeddings with fallback to local models:

- **Primary: OpenAI text-embedding-3-small**

- Dimensions: 1536
- Cost: ~$0.02 per 1M tokens
- Quality: High for technical content
- Speed: Fast via API

- **Fallback: Sentence-Transformers Local Models**
- Models tried in order:
  1. `all-MiniLM-L12-v2` (384 dimensions, better quality)
  2. `all-mpnet-base-v2` (768 dimensions, highest quality, slower)
  3. `all-MiniLM-L6-v2` (384 dimensions, fastest fallback)
- No API costs
- Slower initial model download
- Good for development/testing

### Troubleshooting Embedding Issues

**Dimension Mismatch Errors:**

If you see errors like "Collection expecting embedding with dimension of 384, got 1536":

1. **Delete the old database:**

   ```bash
   # Windows
   Remove-Item -Recurse -Force .ragdb
   
   # Linux/Mac
   rm -rf .ragdb
   ```

2. **Re-run ingestion:**

   ```bash
   python reingest_documents.py
   ```

**Switching Embedding Models:**

To change embedding models, you must recreate the database:

1. Delete existing `.ragdb` folder
2. Update `OPENAI_EMBED_MODEL` in `.env`
3. Run ingestion script again

**Cost Optimization:**

- `text-embedding-3-small`: Best balance of cost/performance
- `text-embedding-3-large`: 2x more expensive, marginal quality gain
- Local models: Free but slower and larger storage requirements

## Using the RAG System

### Accessing the Knowledge Assistant

1. **Start the Application:**

   ```bash
   python run_app.py
   # or
   streamlit run app.py
   ```

2. **Navigate to Report Assistant:**
   - Click on "02_Report_Assistant" in the sidebar
   - Or go directly to the "Knowledge & Report Assistant" page

### Knowledge Base Query (Q&A Mode)

**How to Use:**

1. **Enter Your Question:**

   ```bash
   What are the standard procedures for flutter testing?
   How do I calibrate airspeed indicators?
   What are the safety requirements for spin testing?
   ```

2. **Adjust Retrieval Parameters:**
   - **Number of Results (k)**: 3-10 (more results = more context, slower response)
   - **Database Path**: Usually `.ragdb` (default)

3. **Review Results:**
   - Relevant document excerpts are retrieved
   - Sources are cited with document names and confidence scores
   - Use this information to understand flight test concepts

**Best Practices for Questions:**

- Be specific: "elevator flutter testing procedures" vs. "testing"
- Use technical terms: "angle of attack calibration" vs. "measuring angles"
- Ask one concept per question for focused results
- Reference specific aircraft systems or test types

### Report Generation Mode

**Intelligent Flight Test Reports:**

1. **Upload Flight Data:** Use the main page to upload your CSV flight data
2. **Ask for Report:** Request specific analysis reports:

   ```bash
   Generate a control system analysis report for this flight
   Create a safety assessment based on the recorded parameters
   Analyze the flutter test results and compare to standards
   ```

3. **Context Integration:** The system combines:
   - Your actual flight data and charts
   - Relevant knowledge base content
   - Standard procedures and requirements
   - Safety considerations and recommendations

### Advanced Usage

**Custom Retrieval:**

```python
from components.rag.retrieval import retrieve

# Custom knowledge retrieval
results = retrieve(
    query="flutter testing safety margins",
    k=5,  # Number of relevant chunks
    db_path=".ragdb"
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Source: {result['metadata']['source']}")
    print(f"Content: {result['text'][:200]}...")
    print("-" * 50)
```

## Performance Optimization

### For Large Document Collections

**Chunking Strategy:**

- Default chunk size: 500 characters
- For technical documents: 400-600 characters
- For procedural documents: 300-500 characters
- Overlapping chunks: Consider implementing for continuity

**Batch Processing:**

- Default batch size: 100 documents
- Increase for faster processing with sufficient RAM
- Decrease if experiencing memory issues

**Database Optimization:**

- Regular database maintenance not typically required
- Consider periodic re-indexing for very large collections
- Monitor disk space usage

### Query Performance

**Retrieval Tuning:**

- Start with k=3-5 for focused results
- Increase k=7-10 for comprehensive analysis
- Higher k values increase response time
- Balance relevance vs. response speed

**Response Time Expectations:**

- Simple queries: 1-3 seconds
- Complex queries: 3-8 seconds
- Report generation: 10-30 seconds
- First query after startup: 5-10 seconds (model loading)

## Maintenance and Updates

### Adding New Documents

**Incremental Updates:**

Currently, the system requires full re-ingestion for new documents:

1. Add new documents to `docs/knowledge_base/`
2. Delete existing `.ragdb` folder
3. Run `python reingest_documents.py`

**Future Enhancement:** Incremental document addition without full re-ingestion.

### Database Backup and Recovery

**Backup Strategy:**

```bash
# Create backup
cp -r .ragdb .ragdb_backup_$(date +%Y%m%d)

# Restore from backup
rm -rf .ragdb
cp -r .ragdb_backup_YYYYMMDD .ragdb
```

**Version Control:**

- Add `.ragdb/` to `.gitignore` for development
- Include `.ragdb/` in repository for team sharing
- Consider size limitations for repository storage

### Monitoring and Logging

**Check System Health:**

```python
# Embedding system status
from components.rag.ingest import embed_texts
test_embedding = embed_texts(["test"])
print(f"Embedding dimension: {len(test_embedding[0])}")
print(f"System operational: {len(test_embedding) > 0}")

# Database status
import chromadb
client = chromadb.PersistentClient(path=".ragdb")
coll = client.get_collection("flight_test_kb")
print(f"Document count: {coll.count()}")
```

## Integration with Flight Analysis

### Workflow Integration

**Typical Analysis Session:**

1. **Upload Flight Data** (main page)
2. **Create Analysis Charts** (main page)
3. **Query Knowledge Base** (report assistant page)
   - "What are normal values for this parameter?"
   - "How should I interpret these oscillations?"
4. **Generate Contextual Report** (report assistant page)
   - Combines your data with expert knowledge
   - Provides recommendations and interpretations

### Best Practices for Integration

**Data-Driven Questions:**

```bash
Based on the control surface deflections in my data, what should I check next?
These vibration levels seem high - what are the safety limits?
How do I validate these test results against certification requirements?
```

**Report Enhancement:**

- Use RAG to add context to your technical reports
- Validate findings against established procedures
- Include relevant safety considerations
- Reference applicable standards and requirements

## Troubleshooting

### Common Issues

- **1. "No OpenAI API Key" Error**

- Solution: Add `OPENAI_API_KEY` to `.env` file
- Alternative: System will use local sentence-transformers

- **2. "ChromaDB Collection Not Found"**
- Solution: Run ingestion script to create database
- Check: Ensure `docs/knowledge_base/` contains documents

- **3. "Embedding Dimension Mismatch"**
- Solution: Delete `.ragdb` folder and re-run ingestion
- Cause: Changed embedding models between ingestions

- **4. "No Documents Found"**
- Solution: Check file formats and permissions
- Verify: `docs/knowledge_base/` contains PDF or text files

- **5. Slow Query Response**
- Solution: Reduce retrieval parameter `k`
- Check: Network connection for OpenAI API
- Alternative: Use local embedding models

### Advanced Troubleshooting

**Database Corruption:**

```bash
# Remove corrupted database
rm -rf .ragdb

# Recreate from scratch
python reingest_documents.py
```

**Memory Issues:**

```python
# Reduce batch size in ingest.py
ingest_documents(
    source_dir="docs/knowledge_base",
    db_path=".ragdb",
    batch_size=50  # Reduced from default 100
)
```

**API Rate Limiting:**

- OpenAI free tier: 3 RPM limit
- Paid tiers: Higher limits
- Solution: Add delays between batch processing

## Security and Privacy

### Data Handling

**Local Storage:**

- Vector database stored locally in `.ragdb/`
- Document content never leaves your system except for embedding generation
- Embeddings are mathematical representations, not original text

**API Usage:**

- OpenAI sees document chunks for embedding generation only
- No document content stored on OpenAI servers
- Consider local embedding models for sensitive documents

**Best Practices:**

- Review documents before ingestion for sensitive content
- Use local embedding models for highly confidential material
- Regularly backup your knowledge base and database

### Compliance Considerations

**For Regulated Environments:**

- Verify document usage rights and licenses
- Consider data residency requirements
- Use local embedding models if external API usage is restricted
- Maintain audit logs of document sources and usage

## Next Steps

### After Successful Setup

1. **Test with Sample Query:**

   ```bash
   "What are the key considerations for flight test safety?"
   ```

2. **Upload Sample Flight Data** and generate a report

3. **Explore Advanced Features:**
   - Correlation analysis with knowledge context
   - Multi-parameter reports with expert guidance
   - Safety assessment with regulatory references

### Future Enhancements

**Planned Features:**

- Incremental document updates
- Multi-language support
- Advanced chunking strategies
- Custom embedding fine-tuning
- Integration with external knowledge bases

---

**🎯 Quick Start Checklist:**

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Set up `.env` file with `OPENAI_API_KEY`
- [ ] Place documents in `docs/knowledge_base/`
- [ ] Run `python reingest_documents.py`
- [ ] Test with knowledge query in Report Assistant
- [ ] Generate your first intelligent flight test report

**Need Help?** Check the troubleshooting section or contact your development team with specific error messages and system configuration details.
