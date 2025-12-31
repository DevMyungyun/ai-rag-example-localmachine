# RAG Pipeline: Web Pages to Vector Database

Transform web page content into chunks suitable for vector database insertion and retrieval.

## Features

- ✅ Fetch content from multiple web pages
- ✅ Process HTML using docling for clean text extraction
- ✅ Intelligent text chunking with overlap
- ✅ Automatic embedding generation using Sentence Transformers
- ✅ Store in PostgreSQL with pgvector extension
- ✅ Query interface for retrieval

## Prerequisites

**Python Version**: Python 3.11 or 3.12 is required. Python 3.14+ is **not compatible** with the current versions of chromadb and langchain libraries.

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

1. **Fetch**: Downloads HTML content from specified URLs
2. **Parse**: Uses docling to extract clean text from HTML
3. **Chunk**: Splits content into overlapping chunks for better context
4. **Embed**: Generates vector embeddings using Sentence Transformers
5. **Store**: Saves embeddings and metadata in PostgreSQL with pgvector
6. **Query**: Enables semantic search over stored documents using cosine similarity

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
OLLAMA_MODEL=gemma3:27b

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

## RAG Query System

After ingesting documents, you can query them using Ollama:

### Setup Ollama

```bash
# Start services
docker-compose up -d

# Pull the Ollama model (first time only)
./setup_ollama.sh
# or manually:
docker exec -it ollama-rag ollama pull llama3.2
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

# Show retrieved context
python rag_query.py -q "What is the API endpoint?" -c

# Use a different model
python rag_query.py -m mistral -q "Explain the OAuth flow"

# Retrieve more documents for context
python rag_query.py -q "What are the rate limits?" -n 10
```

### Available Ollama Models

```bash
# List installed models
docker exec -it ollama-rag ollama list

# Pull other models
docker exec -it ollama-rag ollama pull llama3.1
docker exec -it ollama-rag ollama pull mistral
docker exec -it ollama-rag ollama pull codellama
docker exec -it ollama-rag ollama pull phi
```

## Advanced Usage

## Tips for RAG Applications

1. **Chunk Size**: Adjust based on your use case
   - Smaller (300-500): More precise retrieval
   - Larger (1000-2000): More context per chunk

2. **Overlap**: Keep 10-20% of chunk_size for context continuity

3. **Embedding Models**: 
   - `all-MiniLM-L6-v2`: Fast, good balance (default)
   - `all-mpnet-base-v2`: Better quality, slower
   - `multi-qa-mpnet-base-dot-v1`: Optimized for Q&A

4. **Query Optimization**: Rephrase queries for better results

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

**Problem**: `PydanticImportError` or compatibility errors with Python 3.14+

**Solution**: Downgrade to Python 3.11 or 3.12. Python 3.14 is not compatible with current versions of chromadb and langchain:

```bash
pyenv install 3.11.11
pyenv local 3.11.11
pip install -r requirements.txt
```

### Other Issues

- **Memory Issues**: Process URLs in smaller batches
- **Timeout Errors**: Increase timeout in `fetch_web_page()`
- **Poor Retrieval**: Adjust chunk_size or try different embedding model
- **PostgreSQL Connection Error**: Make sure Docker container is running (`docker-compose ps`)
- **Table Creation Error**: Ensure pgvector extension is properly installed in PostgreSQL
