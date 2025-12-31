# RAG Accuracy Improvements Guide

This implementation includes several techniques to improve RAG accuracy:

## 1. **Improved Chunking Strategy**
- Better text separators (sentences, punctuation)
- Optimal chunk size and overlap for context preservation
- Default: 1000 chars with 200 overlap

## 2. **Hybrid Search**
- Combines vector similarity with keyword matching
- Boosts results that contain exact query terms
- Better for specific terms and phrases

## 3. **Query Expansion**
- Automatically generates query variations
- Adds question formats ("How to...", "What is...")
- Searches with multiple query forms

## 4. **Reranking**
- Uses Cross-Encoder model for better relevance scoring
- Retrieves more candidates (3x) then reranks
- Significantly improves result quality
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## 5. **Enhanced Prompts**
- Clearer instructions for the LLM
- Asks for citation of document sources
- Encourages comprehensive answers
- Handles conflicting information

## 6. **Multiple Retrieval Strategies**
- Vector similarity search
- Keyword matching (ILIKE)
- Deduplication of results
- Configurable result count

## Usage Tips for Better Accuracy:

### When Ingesting Documents:
```bash
# Use better chunking parameters
python web_to_vector_db.py
# Default: chunk_size=1000, overlap=200 (good balance)
```

### When Querying:
```bash
# Retrieve more documents for complex questions
python rag_query.py -q "your question" -n 10

# Show context to verify relevance
python rag_query.py -q "your question" -c

# Use better models
python rag_query.py -m llama3.1 -q "your question"
```

### Interactive Mode Commands:
- `context` - Toggle showing retrieved documents
- `more` - Switch between 5 and 10 results
- Be specific in your questions
- Ask follow-up questions for clarification

## Advanced: Use Better Embedding Models

Edit `web_to_vector_db.py` and `rag_query.py`:
```python
# Instead of: 'all-MiniLM-L6-v2' (384 dimensions)
# Use: 'all-mpnet-base-v2' (768 dimensions) - better quality
# Or: 'all-MiniLM-L12-v2' (384 dimensions) - balanced

self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
```

Remember to:
1. Re-ingest all documents with the new model
2. Update embedding_dimension in the code
3. Recreate the PostgreSQL table

## Advanced: Use Better LLM Models

```bash
# Pull larger models
docker exec -it ollama-rag ollama pull llama3.1:70b
docker exec -it ollama-rag ollama pull mixtral
docker exec -it ollama-rag ollama pull qwen2.5

# Use in queries
python rag_query.py -m llama3.1:70b -q "your question"
```

## Troubleshooting Low Accuracy:

1. **Check your data**: `python check_db.py`
2. **Verify document quality**: Are they comprehensive?
3. **Test queries**: Start with simple, then complex
4. **Increase context**: Use `-n 10` or more
5. **Check similarity scores**: Use `-c` to see relevance
6. **Try different models**: Some models handle certain topics better
7. **Adjust chunk size**: Smaller chunks (500) for precise info, larger (1500) for context
