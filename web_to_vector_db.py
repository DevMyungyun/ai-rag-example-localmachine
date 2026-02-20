#!/usr/bin/env python3
"""
RAG Pipeline: Web Pages to Vector Database
Transforms web page content into chunks suitable for vector database insertion.
"""

import os
from typing import List, Dict
from pathlib import Path
import requests
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np

load_dotenv()


class WebToVectorPipeline:
    """Pipeline to process web pages and store in vector database."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        table_name: str = "web_documents",
        db_host: str = None,
        db_port: int = None,
        db_name: str = None,
        db_user: str = None,
        db_password: str = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            table_name: Name of the database table
            db_host: PostgreSQL host (default: from env or localhost)
            db_port: PostgreSQL port (default: from env or 5432)
            db_name: Database name (default: from env or rag_database)
            db_user: Database user (default: from env or postgres)
            db_password: Database password (default: from env or postgres)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_name = table_name
        
        # Get PostgreSQL connection details from env or use defaults
        self.db_host = db_host or os.getenv("POSTGRES_HOST", "localhost")
        self.db_port = db_port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.db_name = db_name or os.getenv("POSTGRES_DB", "rag_database")
        self.db_user = db_user or os.getenv("POSTGRES_USER", "postgres")
        self.db_password = db_password or os.getenv("POSTGRES_PASSWORD", "postgres")
        
        # Initialize document converter (docling)
        self.converter = DocumentConverter()
        
        # Initialize text splitter with improved separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors
        
        # Initialize PostgreSQL connection
        print(f"Connecting to PostgreSQL at {self.db_host}:{self.db_port}")
        self.conn = psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )
        
        # Create table and extension first
        self._create_table()
        
        # Register vector type after extension is created
        register_vector(self.conn)
    
    def _create_table(self) -> None:
        """Create the vector database table if it doesn't exist."""
        with self.conn.cursor() as cur:
            # Enable required extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            self.conn.commit()
            
            # Create table with vector column and tsvector for full-text search
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({self.embedding_dimension}),
                    source TEXT,
                    title TEXT,
                    chunk_index INTEGER,
                    total_chunks INTEGER,
                    doc_type TEXT,
                    section_header TEXT,
                    content_tsvector tsvector,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Get row count for optimal IVFFlat lists calculation
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
            row_count = cur.fetchone()[0]
            lists = max(100, row_count // 1000) if row_count > 0 else 100
            
            # Create index for vector similarity search with dynamic lists
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists});
            """)
            
            # Create GIN index for full-text search on tsvector
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_content_tsvector_idx
                ON {self.table_name}
                USING GIN (content_tsvector);
            """)
            
            # Create GIN trigram index for fuzzy text matching
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_content_trgm_idx
                ON {self.table_name}
                USING GIN (content gin_trgm_ops);
            """)
            
            # Create B-tree index on source for faster URL lookups
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_source_idx
                ON {self.table_name} (source);
            """)
            
            self.conn.commit()
            print(f"Table '{self.table_name}' is ready with optimized indexes")
    
    def check_url_exists(self, url: str) -> int:
        """
        Check if a URL already exists in the database.
        
        Args:
            url: URL to check
            
        Returns:
            Number of existing chunks for this URL
        """
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE source = %s;",
                (url,)
            )
            count = cur.fetchone()[0]
        return count
    
    def delete_url_data(self, url: str) -> int:
        """
        Delete all data for a specific URL.
        
        Args:
            url: URL to delete
            
        Returns:
            Number of rows deleted
        """
        with self.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table_name} WHERE source = %s;",
                (url,)
            )
            deleted = cur.rowcount
            self.conn.commit()
        return deleted
    
    def __del__(self):
        """Clean up database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    @staticmethod
    def load_urls_from_file(filepath: str) -> List[str]:
        """
        Load URLs from a text file.
        
        Args:
            filepath: Path to file containing URLs (one per line)
            
        Returns:
            List of valid URLs
        """
        urls = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        urls.append(line)
            print(f"Loaded {len(urls)} URLs from {filepath}")
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found")
        except Exception as e:
            print(f"Error reading file '{filepath}': {e}")
        
        return urls
    
    @staticmethod
    def check_url_accessible(url: str, timeout: int = 10) -> Dict:
        """
        Check if a URL is accessible.
        
        Args:
            url: URL to check
            timeout: Request timeout in seconds
            
        Returns:
            Dict with status information
        """
        result = {
            'url': url,
            'accessible': False,
            'status_code': None,
            'error': None,
            'redirect_url': None
        }
        
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            result['accessible'] = response.status_code < 400
            result['status_code'] = response.status_code
            
            # Check if redirected
            if response.url != url:
                result['redirect_url'] = response.url
            
            # If HEAD fails, try GET (some servers don't support HEAD)
            if response.status_code >= 400:
                response = requests.get(url, timeout=timeout, stream=True)
                result['accessible'] = response.status_code < 400
                result['status_code'] = response.status_code
                
        except requests.exceptions.Timeout:
            result['error'] = 'Timeout'
        except requests.exceptions.ConnectionError:
            result['error'] = 'Connection Error'
        except requests.exceptions.TooManyRedirects:
            result['error'] = 'Too Many Redirects'
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def validate_urls(self, urls: List[str]) -> List[str]:
        """
        Validate a list of URLs and return only accessible ones.
        
        Args:
            urls: List of URLs to validate
            
        Returns:
            List of accessible URLs
        """
        print(f"\nValidating {len(urls)} URLs...")
        accessible_urls = []
        
        for i, url in enumerate(urls, 1):
            print(f"  [{i}/{len(urls)}] Checking {url}...", end=' ')
            result = self.check_url_accessible(url)
            
            if result['accessible']:
                print(f"✓ OK (status: {result['status_code']})")
                if result['redirect_url']:
                    print(f"      → Redirects to: {result['redirect_url']}")
                accessible_urls.append(url)
            else:
                error_msg = result['error'] or f"Status {result['status_code']}"
                print(f"✗ FAILED ({error_msg})")
        
        print(f"\n✓ {len(accessible_urls)}/{len(urls)} URLs are accessible\n")
        return accessible_urls
    
    def fetch_web_page(self, url: str) -> str:
        """
        Fetch content from a web page.
        
        Args:
            url: URL of the web page
            
        Returns:
            HTML content of the page
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""
    
    def save_temp_html(self, html_content: str, url: str) -> Path:
        """
        Save HTML content to a temporary file.
        
        Args:
            html_content: HTML content to save
            url: Source URL (used for naming)
            
        Returns:
            Path to the temporary file
        """
        temp_dir = Path("./temp_html")
        temp_dir.mkdir(exist_ok=True)
        
        # Create a safe filename from URL
        filename = url.replace("https://", "").replace("http://", "")
        filename = "".join(c if c.isalnum() else "_" for c in filename)
        filepath = temp_dir / f"{filename}.html"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return filepath
    
    def process_web_page(self, url: str, update_if_exists: bool = True) -> List[Document]:
        """
        Process a single web page using docling.
        
        Args:
            url: URL of the web page
            update_if_exists: If True, delete existing data before processing
            
        Returns:
            List of Document objects with extracted content
        """
        print(f"Processing: {url}")
        
        # Check if URL already exists
        if update_if_exists:
            existing_count = self.check_url_exists(url)
            if existing_count > 0:
                print(f"  ⚠️  URL already exists with {existing_count} chunks")
                print(f"  🗑️  Deleting old data...")
                deleted = self.delete_url_data(url)
                print(f"  ✓ Deleted {deleted} old chunks")
        
        # Fetch the web page
        html_content = self.fetch_web_page(url)
        if not html_content:
            return []
        
        # Save to temporary file
        temp_file = self.save_temp_html(html_content, url)
        
        try:
            # Use docling to convert and extract content
            result = self.converter.convert(str(temp_file))
            
            # Extract text content
            text_content = result.document.export_to_markdown()
            
            # Create metadata
            metadata = {
                "source": url,
                "title": getattr(result.document, "title", url),
                "type": "web_page"
            }
            
            # Create document
            doc = Document(
                page_content=text_content,
                metadata=metadata
            )
            
            return [doc]
            
        except Exception as e:
            print(f"Error processing {url} with docling: {e}")
            return []
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk index to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
            
            chunked_docs.extend(chunks)
        
        return chunked_docs
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for text chunks.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def insert_to_vector_db(self, documents: List[Document]) -> None:
        """
        Insert documents into vector database.
        
        Args:
            documents: List of documents to insert
        """
        if not documents:
            print("No documents to insert")
            return
        
        print(f"Inserting {len(documents)} chunks into vector database...")
        
        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Insert into PostgreSQL with tsvector population
        with self.conn.cursor() as cur:
            for doc, embedding in zip(documents, embeddings):
                metadata = doc.metadata
                cur.execute(f"""
                    INSERT INTO {self.table_name} 
                    (content, embedding, source, title, chunk_index, total_chunks, doc_type, section_header, content_tsvector)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('english', %s))
                """, (
                    doc.page_content,
                    embedding,
                    chunk.get('source'),
                    chunk.get('title'),
                    chunk.get('chunk_index'),
                    chunk.get('total_chunks'),
                    chunk.get('type'),
                    chunk.get('section_header', ''),
                    chunk['content']
                ))
        
        self.conn.commit()
        print(f"Successfully inserted {len(documents)} chunks")
    
    def process_urls(self, urls: List[str], update_if_exists: bool = True) -> None:
        """
        Process multiple URLs and insert into vector database.
        
        Args:
            urls: List of URLs to process
            update_if_exists: If True, update existing URLs instead of creating duplicates
        """
        all_documents = []
        
        # Process each URL
        for url in urls:
            docs = self.process_web_page(url, update_if_exists=update_if_exists)
            all_documents.extend(docs)
        
        if not all_documents:
            print("No documents were successfully processed")
            return
        
        # Chunk documents
        print(f"\nChunking {len(all_documents)} documents...")
        chunked_docs = self.chunk_documents(all_documents)
        print(f"Created {len(chunked_docs)} chunks")
        
        # Insert into vector database
        self.insert_to_vector_db(chunked_docs)
    
    def query(self, query_text: str, n_results: int = 5) -> Dict:
        """
        Query the vector database.
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            
        Returns:
            Query results with documents and metadata
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode([query_text])[0].tolist()
        
        # Query PostgreSQL using cosine similarity
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT 
                    content,
                    source,
                    title,
                    chunk_index,
                    total_chunks,
                    doc_type,
                    1 - (embedding <=> %s::vector) as similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, n_results))
            
            results = cur.fetchall()
        
        # Format results similar to ChromaDB output
        documents = []
        metadatas = []
        distances = []
        
        for row in results:
            documents.append(row[0])  # content
            metadatas.append({
                'source': row[1],
                'title': row[2],
                'chunk_index': row[3],
                'total_chunks': row[4],
                'type': row[5]
            })
            distances.append(1 - row[6])  # Convert similarity to distance
        
        return {
            'documents': [documents],
            'metadatas': [metadatas],
            'distances': [distances]
        }


def main():
    """Example usage of the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest web pages into vector database')
    parser.add_argument(
        '--urls-file', '-f',
        type=str,
        default='urls.txt',
        help='Path to file containing URLs (default: urls.txt)'
    )
    parser.add_argument(
        '--skip-validation', '-s',
        action='store_true',
        help='Skip URL validation (faster but may process broken links)'
    )
    parser.add_argument(
        '--chunk-size', '-c',
        type=int,
        default=1000,
        help='Chunk size for text splitting (default: 1000)'
    )
    parser.add_argument(
        '--chunk-overlap', '-o',
        type=int,
        default=200,
        help='Chunk overlap (default: 200)'
    )
    parser.add_argument(
        '--force', '-F',
        action='store_true',
        help='Force reprocess URLs even if they exist (updates existing data)'
    )
    parser.add_argument(
        '--append', '-a',
        action='store_true',
        help='Append to existing data (create duplicates) instead of updating'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = WebToVectorPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        table_name="web_documents"
    )
    
    # Load URLs from file
    urls = pipeline.load_urls_from_file(args.urls_file)
    
    if not urls:
        print("No URLs to process. Add URLs to", args.urls_file)
        return
    
    # Validate URLs unless skipped
    if not args.skip_validation:
        urls = pipeline.validate_urls(urls)
        
        if not urls:
            print("No accessible URLs found. Check your URLs and try again.")
            return
    
    # Determine update behavior
    update_mode = not args.append  # By default, update existing data
    
    if update_mode:
        print("\n📝 Mode: UPDATE (will replace existing URLs)")
    else:
        print("\n📝 Mode: APPEND (will create duplicates)")
    
    # Process URLs and insert into vector database
    pipeline.process_urls(urls, update_if_exists=update_mode)
    
    # Show final statistics
    print("\n" + "="*70)
    print("📊 Database Statistics")
    print("="*70)
    
    with pipeline.conn.cursor() as cur:
        # Total chunks
        cur.execute(f"SELECT COUNT(*) FROM {pipeline.table_name};")
        total_chunks = cur.fetchone()[0]
        print(f"Total chunks in database: {total_chunks}")
        
        # Unique sources
        cur.execute(f"SELECT COUNT(DISTINCT source) FROM {pipeline.table_name};")
        unique_sources = cur.fetchone()[0]
        print(f"Unique sources: {unique_sources}")
        
        # Show sources
        cur.execute(f"""
            SELECT source, COUNT(*) as chunk_count
            FROM {pipeline.table_name}
            GROUP BY source
            ORDER BY chunk_count DESC;
        """)
        
        print("\nSources in database:")
        for source, count in cur.fetchall():
            print(f"  - {source}: {count} chunks")
    
    print("="*70)


if __name__ == "__main__":
    main()
