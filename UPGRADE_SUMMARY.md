# RAG System Upgrade Summary

## Changes Made to Improve Response Accuracy

### 1. **Updated Dependencies** ([requirements.txt](requirements.txt))

**Removed:**
- `langchain-text-splitters` - Replaced with more sophisticated chunking
- `langchain-core` - No longer needed without langchain splitters

**Added:**
- `chonkie` - For semantic-based chunking (chunks by meaning, not just character count)
- `transformers` - Required by Chonkie for semantic analysis

**Retained:**
- `docling` - Enhanced usage for better HTML to Markdown conversion
- All other dependencies remain the same

### 2. **Enhanced Web Scraping & Document Processing** ([web_to_vector_db.py](web_to_vector_db.py))

#### HTML Cleaning (New Method: `clean_markdown_content`)
- **Strips non-substantive content:** Navigation menus, footers, headers, ads, social media links
- **Removes cookie notices and newsletter prompts**
- **Filters out short navigation-like lines** while preserving markdown headers
- **Eliminates excessive whitespace** for cleaner text

#### Section Header Extraction (New Methods: `extract_section_headers`, `get_section_for_position`)
- **Extracts markdown headers** (# Header) with their positions in the document
- **Associates each chunk with its section** for better context and traceability
- **Preserves document structure** in metadata

#### Database Schema Update
- Added `section_header` column to store which section each chunk belongs to
- Improves traceability: know exactly which part of the document each chunk came from

### 3. **Semantic Chunking with Chonkie** ([web_to_vector_db.py](web_to_vector_db.py))

**Replaced:** `RecursiveCharacterTextSplitter` (LangChain)  
**With:** `SemanticChunker` (Chonkie)

#### Key Improvements:
- **Chunks based on semantic meaning** rather than arbitrary character counts
- **Uses the same embedding model** (`all-MiniLM-L6-v2`) for consistent semantic understanding
- **Intelligent boundary detection** - breaks at natural semantic boundaries

#### Configuration:
- **Chunk size:** 500-1,500 characters (default: 1000)
- **Chunk overlap:** 50-150 tokens (default: 100) - reduced from 200 for better performance
- **Threshold:** 0.5 semantic similarity for chunk boundaries
- **Minimum sentences:** 1 (ensures chunks are at least one complete sentence)

#### Metadata Preservation:
Each chunk now includes:
- `source` - Original URL
- `title` - Document title
- `chunk_index` - Position in document
- `total_chunks` - Total number of chunks from this document
- `section_header` - Which section this chunk belongs to
- `type` - Document type (web_page)

### 4. **Updated Processing Pipeline**

#### Old Flow:
1. Fetch HTML
2. Convert to markdown with Docling
3. Split with RecursiveCharacterTextSplitter
4. Create embeddings and store

#### New Flow:
1. Fetch HTML
2. Convert to markdown with Docling
3. **Clean markdown** (remove navigation, ads, footers)
4. **Extract section headers** for context
5. **Semantic chunking** with Chonkie (meaning-based)
6. **Associate chunks with sections**
7. Create embeddings and store with enhanced metadata

## How This Improves RAG Accuracy

### 1. **Cleaner Input Data**
- Removes noise (navigation, ads) that could confuse the retrieval system
- Only substantive content is indexed
- Better signal-to-noise ratio in embeddings

### 2. **Semantic Coherence**
- Chunks respect natural semantic boundaries
- Related concepts stay together
- Reduces mid-sentence or mid-thought breaks
- More contextually complete chunks for better matching

### 3. **Enhanced Metadata**
- Section headers provide additional context
- Better traceability for answers
- Enables more precise source citation
- Helps LLM understand document structure

### 4. **Optimal Chunk Size**
- 500-1,500 character range balances:
  - **Enough context** for semantic understanding
  - **Focused content** for precise retrieval
  - **Efficient processing** and storage

### 5. **Reduced Overlap**
- Changed from 200 to 100 token overlap
- Less redundancy = more diverse context retrieved
- Still maintains continuity across chunk boundaries

## Installation & Usage

### Step 1: Install Updated Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Reingest Your Documents

Since the database schema changed (added `section_header` column), you should either:

**Option A: Drop and recreate** (if you want fresh data)
```bash
# Clear existing data
python manage_db.py

# Reingest with new pipeline
python web_to_vector_db.py
```

**Option B: Add column to existing database**
```sql
ALTER TABLE web_documents ADD COLUMN section_header TEXT;
```

Then reingest documents to populate the new field:
```bash
python web_to_vector_db.py --force
```

### Step 3: Query as Normal

The query system ([rag_query.py](rag_query.py)) will automatically benefit from the improved chunks:

```bash
# Interactive mode
python rag_query.py

# Single question
python rag_query.py -q "Your question here"
```

## Expected Improvements

1. **More Relevant Retrieval** - Semantic chunking means chunks are more contextually complete
2. **Better Answers** - Cleaner input → better embeddings → more accurate retrieval
3. **Improved Citations** - Section headers help identify exactly where information came from
4. **Reduced Hallucination** - Less noise in the corpus means less chance of retrieving irrelevant content

## Advanced Configuration

You can tune the chunking parameters:

```bash
# Smaller chunks (more precise, less context)
python web_to_vector_db.py --chunk-size 500 --chunk-overlap 50

# Larger chunks (more context, less precise)
python web_to_vector_db.py --chunk-size 1500 --chunk-overlap 150
```

**Recommendation:** Start with defaults (1000/100) and adjust based on your specific use case and document types.

## Technical Notes

### Chonkie Semantic Chunker
- Uses embedding model to understand semantic similarity between sentences
- Groups sentences that are semantically related
- Respects the specified chunk_size as a target, not a hard limit
- May produce slightly variable chunk sizes for optimal semantic coherence

### Section Header Tracking
- Headers are extracted using regex pattern: `^(#{1,6})\s+(.+)$`
- Position tracking allows accurate association of chunks with sections
- Empty section_header indicates content before the first header

## Troubleshooting

### If you get import errors:
```bash
pip install --upgrade -r requirements.txt
```

### If semantic chunking is too slow:
- Consider using a smaller embedding model (though this may reduce quality)
- Or increase chunk_size to create fewer chunks

### If chunks are too large/small:
- Adjust `--chunk-size` parameter
- Chonkie will try to respect this while maintaining semantic coherence

## References

- **Chonkie**: Python library for semantic chunking
- **Docling**: IBM's document conversion library
- **Semantic Chunking Strategy**: Inspired by modern RAG best practices (Chatfluence project reference)
