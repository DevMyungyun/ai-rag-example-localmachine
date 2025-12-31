#!/usr/bin/env python3
"""
RAG Query System with Ollama
Ask questions and get answers based on your vector database documents.
"""

import os
import sys
from typing import List, Dict, Tuple
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
import ollama
import re

load_dotenv()


class RAGQuerySystem:
    """RAG system that retrieves context and generates answers using Ollama."""
    
    def __init__(
        self,
        table_name: str = "web_documents",
        db_host: str = None,
        db_port: int = None,
        db_name: str = None,
        db_user: str = None,
        db_password: str = None,
        ollama_model: str = None,
        ollama_host: str = None
    ):
        """
        Initialize the RAG query system.
        
        Args:
            table_name: Name of the database table
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password
            ollama_model: Ollama model to use (default: llama3.2)
            ollama_host: Ollama host URL
        """
        self.table_name = table_name
        
        # Get PostgreSQL connection details
        self.db_host = db_host or os.getenv("POSTGRES_HOST", "localhost")
        self.db_port = db_port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.db_name = db_name or os.getenv("POSTGRES_DB", "rag_database")
        self.db_user = db_user or os.getenv("POSTGRES_USER", "postgres")
        self.db_password = db_password or os.getenv("POSTGRES_PASSWORD", "postgres")
        
        # Ollama settings
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2")
        ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Initialize embedding model (same as used for document ingestion)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize reranker for better results
        print("Loading reranker model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize PostgreSQL connection
        print(f"Connecting to PostgreSQL at {self.db_host}:{self.db_port}")
        self.conn = psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )
        register_vector(self.conn)
        
        # Set Ollama client host
        ollama._client._client.base_url = ollama_host
        print(f"Using Ollama model: {self.ollama_model} at {ollama_host}")
        
        # Check if model is available
        self._check_ollama_model()
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand the query with synonyms and related terms.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        # Basic query expansion - add variations
        expanded = [query]
        
        # Add common variations
        words = query.lower().split()
        
        # Add question variations
        if not any(q in query.lower() for q in ['how', 'what', 'when', 'where', 'why', 'who']):
            expanded.append(f"How to {query}")
            expanded.append(f"What is {query}")
        
        return expanded
    
    def _check_ollama_model(self) -> None:
        """Check if the Ollama model is available, if not provide instructions."""
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            if not any(self.ollama_model in name for name in model_names):
                print(f"\n⚠️  Warning: Model '{self.ollama_model}' not found!")
                print(f"Available models: {', '.join(model_names) if model_names else 'None'}")
                print(f"\nTo pull the model, run:")
                print(f"  docker exec -it ollama-rag ollama pull {self.ollama_model}")
                print(f"  or: ollama pull {self.ollama_model}\n")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running (docker-compose up -d)")
    
    def retrieve_context(
        self, 
        query: str, 
        n_results: int = 5,
        similarity_threshold: float = 0.0,
        use_reranking: bool = True,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant documents from the vector database with enhanced accuracy.
        
        Args:
            query: User's question
            n_results: Number of documents to retrieve
            similarity_threshold: Minimum similarity score (0-1)
            use_reranking: Whether to use reranking for better results
            use_hybrid: Whether to use hybrid search (vector + keyword)
            
        Returns:
            List of relevant documents with metadata
        """
        # First, check if there are any documents
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
            total_docs = cur.fetchone()[0]
            
            if total_docs == 0:
                print(f"⚠️  Database table '{self.table_name}' is empty!")
                print("Run: python web_to_vector_db.py to ingest documents first.")
                return []
            
            print(f"   Database contains {total_docs} document chunks")
        
        # Expand query for better matching
        expanded_queries = self.expand_query(query)
        print(f"   Searching with {len(expanded_queries)} query variations")
        
        all_results = set()
        
        # Vector search with expanded queries
        for expanded_query in expanded_queries:
            query_embedding = self.embedding_model.encode([expanded_query])[0].tolist()
            
            with self.conn.cursor() as cur:
                # Retrieve more candidates for reranking
                limit = n_results * 3 if use_reranking else n_results
                
                if use_hybrid:
                    # Hybrid search: combine vector similarity with keyword matching
                    cur.execute(f"""
                        SELECT DISTINCT
                            content,
                            source,
                            title,
                            chunk_index,
                            total_chunks,
                            (1 - (embedding <=> %s::vector)) + 
                            CASE 
                                WHEN content ILIKE %s THEN 0.2
                                ELSE 0.0
                            END as similarity
                        FROM {self.table_name}
                        WHERE (1 - (embedding <=> %s::vector)) > %s
                           OR content ILIKE %s
                        ORDER BY similarity DESC
                        LIMIT %s
                    """, (
                        query_embedding, 
                        f'%{query}%',
                        query_embedding, 
                        similarity_threshold,
                        f'%{query}%',
                        limit
                    ))
                else:
                    # Pure vector search
                    cur.execute(f"""
                        SELECT 
                            content,
                            source,
                            title,
                            chunk_index,
                            total_chunks,
                            1 - (embedding <=> %s::vector) as similarity
                        FROM {self.table_name}
                        WHERE 1 - (embedding <=> %s::vector) > %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, query_embedding, similarity_threshold, query_embedding, limit))
                
                results = cur.fetchall()
                all_results.update(results)
        
        # Convert to list and deduplicate
        unique_results = list(all_results)
        
        # Format documents
        documents = []
        for row in unique_results:
            documents.append({
                'content': row[0],
                'source': row[1],
                'title': row[2],
                'chunk_index': row[3],
                'total_chunks': row[4],
                'similarity': round(row[5], 3)
            })
        
        # Rerank results if enabled
        if use_reranking and documents:
            print(f"   Reranking {len(documents)} candidates...")
            
            # Prepare pairs for reranking
            pairs = [[query, doc['content']] for doc in documents]
            
            # Get reranking scores
            scores = self.reranker.predict(pairs)
            
            # Add rerank scores and sort
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)
            
            documents.sort(key=lambda x: x['rerank_score'], reverse=True)
            documents = documents[:n_results]
            
            print(f"   Selected top {len(documents)} after reranking")
        else:
            documents = documents[:n_results]
        
        return documents
    
    def build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """
        Build a prompt for the LLM with context and query.
        
        Args:
            query: User's question
            context_docs: Retrieved documents
            
        Returns:
            Formatted prompt string
        """
        if not context_docs:
            return f"""You are a helpful assistant. The user asked a question but no relevant 
information was found in the knowledge base. Politely inform them that you don't have 
information about this topic.

Question: {query}

Answer:"""
        
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            # Include rerank score if available
            score_info = f"similarity: {doc['similarity']}"
            if 'rerank_score' in doc:
                score_info += f", relevance: {doc['rerank_score']:.3f}"
            
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {doc['source']}\n"
                f"Score: {score_info}\n"
                f"Content: {doc['content']}\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are a knowledgeable assistant. Your task is to answer the user's question based ONLY on the provided context documents.

Instructions:
1. Read all the context documents carefully
2. Provide a comprehensive and accurate answer
3. Always cite the specific document numbers you used (e.g., "According to Document 1...")
4. If the context doesn't contain enough information, clearly state what's missing
5. Combine information from multiple documents when relevant
6. Be specific and detailed in your answer
7. If there are conflicting information, mention both perspectives

Context Documents:
{context}

User Question: {query}

Detailed Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str, stream: bool = False) -> str:
        """
        Generate an answer using Ollama.
        
        Args:
            prompt: The complete prompt with context
            stream: Whether to stream the response
            
        Returns:
            Generated answer
        """
        try:
            if stream:
                print("\nAnswer: ", end="", flush=True)
                response_text = ""
                for chunk in ollama.chat(
                    model=self.ollama_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=True
                ):
                    text = chunk['message']['content']
                    print(text, end="", flush=True)
                    response_text += text
                print()  # New line after streaming
                return response_text
            else:
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                return response['message']['content']
        except Exception as e:
            return f"Error generating answer: {e}\nMake sure Ollama is running and the model is available."
    
    def query(
        self, 
        question: str, 
        n_results: int = 5,
        show_context: bool = False,
        stream: bool = True,
        use_reranking: bool = True
    ) -> Tuple[str, List[Dict]]:
        """
        Complete RAG query: retrieve context and generate answer.
        
        Args:
            question: User's question
            n_results: Number of documents to retrieve
            show_context: Whether to display retrieved context
            stream: Whether to stream the LLM response
            use_reranking: Whether to use reranking for better accuracy
            
        Returns:
            Tuple of (answer, context_documents)
        """
        print(f"\n🔍 Searching for relevant information...")
        
        # Step 1: Retrieve relevant documents with enhanced methods
        context_docs = self.retrieve_context(
            question, 
            n_results, 
            use_reranking=use_reranking,
            use_hybrid=True
        )
        
        if not context_docs:
            print("⚠️  No relevant documents found in the database.")
            answer = "I don't have any information about that in my knowledge base."
            return answer, []
        
        print(f"✓ Found {len(context_docs)} relevant documents")
        
        # Show context if requested
        if show_context:
            print("\n📄 Retrieved Context:")
            for i, doc in enumerate(context_docs, 1):
                print(f"\n{i}. Source: {doc['source']}")
                print(f"   Similarity: {doc['similarity']}")
                print(f"   Content: {doc['content'][:200]}...")
        
        # Step 2: Build prompt
        prompt = self.build_prompt(question, context_docs)
        
        # Step 3: Generate answer with Ollama
        print(f"\n🤖 Generating answer with {self.ollama_model}...")
        answer = self.generate_answer(prompt, stream=stream)
        
        return answer, context_docs
    
    def interactive_mode(self):
        """Run an interactive Q&A session."""
        print("\n" + "="*70)
        print("RAG Query System - Interactive Mode")
        print("="*70)
        print(f"Using model: {self.ollama_model}")
        print(f"Database: {self.table_name}")
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - Type 'context' to toggle showing retrieved documents")
        print("  - Type 'more' to retrieve more documents (default: 5)")
        print("  - Type 'exit' or 'quit' to stop")
        print("="*70 + "\n")
        
        show_context = False
        n_results = 5
        
        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'context':
                    show_context = not show_context
                    status = "enabled" if show_context else "disabled"
                    print(f"Context display {status}")
                    continue
                
                if user_input.lower() == 'more':
                    n_results = 10 if n_results == 5 else 5
                    print(f"Retrieving {n_results} documents per query")
                    continue
                
                # Process the query
                answer, context_docs = self.query(
                    user_input,
                    n_results=n_results,
                    show_context=show_context,
                    stream=True,
                    use_reranking=True
                )
                
                # Show sources
                if context_docs:
                    print("\n📚 Sources:")
                    sources = list(set([doc['source'] for doc in context_docs]))
                    for source in sources:
                        print(f"  - {source}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def __del__(self):
        """Clean up database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


def main():
    """Run the RAG query system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query your RAG system")
    parser.add_argument(
        '--question', '-q',
        type=str,
        help='Ask a single question'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='llama3.2',
        help='Ollama model to use (default: llama3.2)'
    )
    parser.add_argument(
        '--context', '-c',
        action='store_true',
        help='Show retrieved context'
    )
    parser.add_argument(
        '--results', '-n',
        type=int,
        default=5,
        help='Number of documents to retrieve (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = RAGQuerySystem(ollama_model=args.model)
    
    if args.question:
        # Single question mode
        answer, context_docs = rag.query(
            args.question,
            n_results=args.results,
            show_context=args.context,
            stream=True
        )
        
        # Show sources
        if context_docs:
            print("\n📚 Sources:")
            sources = list(set([doc['source'] for doc in context_docs]))
            for source in sources:
                print(f"  - {source}")
    else:
        # Interactive mode
        rag.interactive_mode()


if __name__ == "__main__":
    main()
