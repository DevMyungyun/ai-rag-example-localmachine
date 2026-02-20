# RAG Pipeline: Web Pages to Vector Database

Transform web page content into chunks suitable for vector database insertion and retrieval.

## Features

- ✅ Fetch content from multiple web pages
- ✅ Process HTML using docling for clean text extraction
- ✅ Semantic chunking with Chonkie for better context preservation
- ✅ Automatic embedding generation using Sentence Transformers
- ✅ Store in PostgreSQL with pgvector extension
- ✅ **PostgreSQL Full-Text Search** (tsvector) for fast text ranking
- ✅ **Hybrid Search**: 70% vector similarity + 30% text rank
- ✅ **Reciprocal Rank Fusion (RRF)** reranking for better accuracy
- ✅ Query optimization with embedding cache
- ✅ Production-ready indexes (IVFFlat, GIN, trigram)
- ✅ Comprehensive database management utilities

## Prerequisites

**Python Version**: Python 3.11 or 3.12 is recommended for best compatibility.

If you're using pyenv:

```bash
# Install Python 3.11 with lzma support (required by docling)
export LDFLAGS="-L$(brew --prefix xz)/lib"
export CPPFLAGS="-I$(brew --prefix xz)/include"
pyenv install 3.11.11

# Set it for this project
pyenv local 3.11.11

# Verify installation
python --version  # Should show Python 3.11.11
```

## Installation

### 1. Start PostgreSQL with pgvector in Docker

Using Docker Compose (recommended):

```bash
# Start PostgreSQL
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f postgres
```

Or using Docker directly:

```bash
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_database \
  -v $(pwd)/postgres_data:/var/lib/postgresql/data \
  ankane/pgvector:latest
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your URLs

Edit `urls.txt` and add the web pages you want to process:

```txt
# Add your URLs here (one per line)
https://example.com/page1
https://example.com/page2
https://docs.example.com/api
```

Lines starting with `#` are treated as comments.

### 4. Migrate Existing Databases (If Upgrading)

If you have an existing database from before the full-text search update, run the migration:

```bash
# Check if migration is needed
python migrate_to_fts.py --check

# Run the migration
python migrate_to_fts.py

# Verify indexes were created
python manage_db.py index-stats
```

New installations automatically include all optimizations.

## Usage

### Customization

#### Adjust Chunking Parameters

```python
pipeline = WebToVectorPipeline(
    chunk_size=500,      # Smaller chunks
    chunk_overlap=100,   # Less overlap
)
```

#### Use Different Embedding Model

```python
from sentence_transformers import SentenceTransformer

pipeline = WebToVectorPipeline()
pipeline.embedding_model = SentenceTransformer('all-mpnet-base-v2')
```

## How It Works

### Document Ingestion:
1. **Fetch**: Downloads HTML content from specified URLs
2. **Parse**: Uses docling to extract clean text from HTML
3. **Chunk**: Semantic chunking with Chonkie for natural boundaries
4. **Embed**: Generates vector embeddings using Sentence Transformers
5. **Index**: Creates tsvector for full-text search using PostgreSQL
6. **Store**: Saves embeddings, tsvector, and metadata with optimized indexes

### Query Processing:
1. **Hybrid Search**: Combines vector similarity (70%) + text rank (30%)
2. **Retrieve**: Gets 3x candidates using optimized PostgreSQL queries
3. **Rerank**: Cross-Encoder model scores query-document relevance
4. **RRF**: Reciprocal Rank Fusion combines all ranking signals
5. **Generate**: Ollama LLM generates answer with citations

### Performance:
- **Fast**: GIN indexes eliminate full table scans (<100ms queries)
- **Accurate**: 4 ranking signals (vector, text, hybrid, rerank)
- **Scalable**: Handles 100k+ documents efficiently
- **Transparent**: All scores visible for debugging

## Configuration

### Environment Variables

Create a `.env` file (optional):

```
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

```

### Vector Database

The pipeline connects to PostgreSQL with pgvector extension running in Docker. Make sure the Docker container is running before executing the script.

```bash
# Check if PostgreSQL is running
docker-compose ps

# Connect to PostgreSQL
psql -h localhost -U postgres -d rag_database

# List tables in database
\dt

# Stop PostgreSQL
docker-compose down

# Stop and remove data
docker-compose down -v
```

Data is persisted in the `postgres_data/` directory, so your vector database will survive container restarts.

## Example

### Ingest Documents

```bash
# Basic usage (updates existing URLs automatically)
python web_to_vector_db.py

# Use a different URLs file
python web_to_vector_db.py -f my_urls.txt

# Append mode (creates duplicates instead of updating)
python web_to_vector_db.py -a

# Skip URL validation (faster, but may fail on broken links)
python web_to_vector_db.py -s

# Custom chunking parameters
python web_to_vector_db.py -c 1500 -o 300

# Combine options
python web_to_vector_db.py -f my_urls.txt -s -c 800 -o 150

# Show all options
python web_to_vector_db.py --help
```

The script will:
1. Load URLs from the file
2. Validate each URL is accessible (unless skipped)
3. Check if URL already exists in database
4. Delete old data for existing URLs (unless --append mode)
5. Process and chunk the content
6. Store in PostgreSQL with embeddings
7. Show statistics summary

**Default Behavior:** URLs are **updated** (old data deleted, new data inserted). Use `-a` to append instead.

### Manage Database

```bash
# View database statistics and migration status
python manage_db.py stats

# View index usage and sizes
python manage_db.py index-stats

# Rebuild vector index with optimal parameters
python manage_db.py reindex

# Update query planner statistics (after large ingestion)
python manage_db.py analyze

# Delete documents by source URL pattern
python manage_db.py delete --source "example.com"

# Reset entire database
python manage_db.py reset
```

**When to run ANALYZE:**
- After ingesting large batches of documents
- When query performance degrades
- After reindexing

**Database Optimization:**
The system automatically creates optimized indexes:
- **IVFFlat index**: Vector similarity with dynamic `lists` parameter
- **GIN index**: Full-text search on tsvector
- **GIN trigram index**: Fuzzy text matching
- **B-tree index**: Fast source URL lookups

## RAG Query System

After ingesting documents, you can query them using Ollama with advanced hybrid search and reranking:

### Setup Ollama

```bash
# Start services
docker-compose up -d

# Pull the Ollama model (first time only)
./setup_ollama.sh
# or manually:
docker exec -it ollama-rag ollama pull qwen2.5:7b
```

### Query Your Documents

**Interactive Mode (Recommended):**
```bash
python rag_query.py
```

This starts an interactive chat where you can:
- Ask questions about your documents
- Type `context` to toggle showing retrieved documents
- Type `exit` to quit

**Single Question Mode:**
```bash
# Ask a single question
python rag_query.py -q "How do I authenticate?"

# Show retrieved context with all scores (vector, text, hybrid, rerank, final)
python rag_query.py -q "What is the API endpoint?" -c

# Use a different model
python rag_query.py -m gemma2:9b -q "Explain the OAuth flow"

# Retrieve more documents for context
python rag_query.py -q "What are the rate limits?" -n 10

# List available models
python rag_query.py --list-models
```

**Understanding Scores** (with `-c` flag):
```
Scores: vector: 0.872, text: 0.0234, hybrid: 0.617, relevance: 2.456, combined: 0.033
```
- **vector**: Semantic similarity (0-1, higher is better)
- **text**: PostgreSQL FTS rank (higher is better)
- **hybrid**: Weighted combination (0.7×vector + 0.3×text_normalized)
- **relevance**: Cross-encoder reranking score
- **combined**: Final RRF score (sorts results)

Documents are ranked by the **combined** score for best accuracy.

### Available Ollama Models

```bash
# List installed models
docker exec -it ollama-rag ollama list

# Recommended small models for RAG
docker exec -it ollama-rag ollama pull qwen2.5:7b      # Best for RAG citations
docker exec -it ollama-rag ollama pull gemma2:9b       # Great reasoning
docker exec -it ollama-rag ollama pull llama3.2:3b     # Fastest, smallest

# Other popular models
docker exec -it ollama-rag ollama pull mistral
docker exec -it ollama-rag ollama pull codellama
docker exec -it ollama-rag ollama pull phi3:mini
```

**Recommended for RAG:**
- **qwen2.5:7b** (4.3GB): Best at following citation instructions
- **gemma2:9b** (5.4GB): Excellent reasoning, good balance
- **llama3.2:3b** (2GB): Fast testing, decent quality

### Memory Requirements for Large Models

**Important**: Large language models require significant memory. If you encounter memory errors like:

```
Error: 500 Internal Server Error: model requires more system memory (18.4 GiB) than is available (6.6 GiB)
```

**Solutions:**

1. **Increase Docker Memory Allocation** (Recommended for Mac/Windows):
   - Open Docker Desktop
   - Go to Settings → Resources
   - Increase Memory to at least 20GB for large models
   - Click "Apply & Restart"

2. **Use Smaller Models**:
   ```bash
   # Use recommended smaller models (ordered by size):
   python rag_query.py -m llama3.2:3b     # ~2GB
   python rag_query.py -m mistral         # ~4GB
   python rag_query.py -m qwen2.5:7b      # ~4.3GB (default, best for RAG)
   python rag_query.py -m gemma2:9b       # ~5.4GB
   ```

3. **Check Available Models and Their Memory Requirements**:
   ```bash
   # List models with sizes
   python rag_query.py --list-models
   
   # Test if a model can run
   ollama run <model-name> "test"
   ```

**Memory Requirements by Model Size:**
- 3B parameter models: ~4-6 GB RAM
- 7B parameter models: ~8-12 GB RAM
- 13B parameter models: ~16-20 GB RAM
- 27B+ parameter models: ~20-32 GB RAM

**Recommended for Development:**
- Default Docker allocation: 8GB → Use models up to 7B parameters
- Production/Large models: 20GB+ → Can use 13B-27B parameter models

## Performance & Accuracy

### Query Performance:
- **Small databases** (<10k docs): <50ms average
- **Medium databases** (10k-100k docs): <100ms average  
- **Large databases** (>100k docs): <200ms average

### Accuracy Features:
1. **Hybrid Search**: Combines semantic and lexical matching
2. **Query Expansion**: Generates variations for better recall
3. **Reranking**: Cross-Encoder improves precision
4. **RRF**: Robust score fusion from multiple signals
5. **Embedding Cache**: Speeds up repeated queries

### Best Practices:
- Run `python manage_db.py analyze` after large ingestion
- Use `-n 10` for complex questions needing more context
- Enable `-c` flag during development to inspect scores
- Check `python manage_db.py index-stats` for index health

For detailed accuracy tuning, see [ACCURACY_GUIDE.md](ACCURACY_GUIDE.md).

## Tips for RAG Applications

1. **Chunk Size**: Adjust based on your use case
   - Smaller (500-800): More precise retrieval, better for facts
   - Larger (1000-1500): More context per chunk, better for explanations
   - Default (1000): Good balance for most use cases

2. **Overlap**: Keep 10-15% of chunk_size for context continuity

3. **Embedding Models**: 
   - `all-MiniLM-L6-v2`: Fast, good balance (default, 384 dims)
   - `bge-small-en-v1.5`: Better quality, same size (384 dims)
   - `nomic-embed-text`: Excellent for RAG (768 dims)
   - `all-mpnet-base-v2`: Higher quality, slower (768 dims)
   
   **Note**: Changing embedding model requires re-ingesting all documents

4. **Query Optimization**: 
   - Start with simple queries to test retrieval
   - Use `-c` flag to inspect which documents are retrieved
   - Increase `-n` for complex questions needing more context
   - Check scores to understand ranking decisions

5. **Maintenance**:
   - Run `analyze` after large ingestion batches
   - Monitor index usage with `index-stats`
   - Reindex vector store if adding many documents

6. **Production Tips**:
   - Use `qwen2.5:7b` or `gemma2:9b` for better answers
   - Allocate sufficient Docker memory (8GB minimum, 20GB for large models)
   - Enable full-text search migration for existing databases
   - Monitor query performance and optimize as needed

## Troubleshooting

### Python Compatibility Issues

**Problem**: `ModuleNotFoundError: No module named '_lzma'`

**Solution**: Python was compiled without lzma support. Reinstall Python with proper flags:

```bash
export LDFLAGS="-L$(brew --prefix xz)/lib"
export CPPFLAGS="-I$(brew --prefix xz)/include"
pyenv uninstall -f 3.11.11
pyenv install 3.11.11
pip install -r requirements.txt
```

**Problem**: `PydanticImportError` or compatibility errors with Python 3.13+

**Solution**: Use Python 3.11 or 3.12 for best compatibility:

```bash
pyenv install 3.11.11
pyenv local 3.11.11
pip install -r requirements.txt
```

### Database Issues

- **"content_tsvector column not found"**: Run migration: `python migrate_to_fts.py`
- **Slow queries**: Run `python manage_db.py analyze` to update statistics
- **PostgreSQL Connection Error**: Make sure Docker container is running (`docker-compose ps`)
- **Table Creation Error**: Ensure pgvector extension is properly installed

### Performance Issues

- **Memory Issues**: Process URLs in smaller batches
- **Timeout Errors**: Increase timeout in `fetch_web_page()`
- **Poor Retrieval**: 
  - Check if migration was run: `python migrate_to_fts.py --check`
  - Inspect scores with `-c` flag
  - Try adjusting chunk_size or embedding model
  - Increase number of results with `-n 10`
- **Index not being used**: Run `python manage_db.py reindex` then `analyze`

### Query Issues

- **No relevant results**: 
  - Verify documents were ingested: `python manage_db.py stats`
  - Check FTS is enabled: Look for tsvector status in stats output
  - Try broader or more specific queries
  - Use `-c` to see what's being retrieved
- **Wrong results ranked highly**: 
  - Reranking should help - ensure it's enabled (default)
  - Check individual scores to see which component needs tuning
  - See [ACCURACY_GUIDE.md](ACCURACY_GUIDE.md) for detailed troubleshooting
