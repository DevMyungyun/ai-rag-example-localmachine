#!/usr/bin/env python3
"""
Check database contents and statistics
"""

import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to database
conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432"),
    database=os.getenv("POSTGRES_DB", "rag_database"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres")
)

cursor = conn.cursor()

# Check if table exists
cursor.execute("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = 'web_documents'
    );
""")
table_exists = cursor.fetchone()[0]

if not table_exists:
    print("❌ Table 'web_documents' does not exist!")
    print("Run: python web_to_vector_db.py to create and populate it.")
else:
    print("✓ Table 'web_documents' exists")
    
    # Count documents
    cursor.execute("SELECT COUNT(*) FROM web_documents;")
    count = cursor.fetchone()[0]
    
    print(f"✓ Total documents: {count}")
    
    if count == 0:
        print("\n❌ No documents in database!")
        print("Run: python web_to_vector_db.py to ingest documents.")
    else:
        # Show sample documents
        cursor.execute("""
            SELECT source, title, content 
            FROM web_documents 
            LIMIT 3;
        """)
        
        print("\n📄 Sample documents:")
        for i, (source, title, content) in enumerate(cursor.fetchall(), 1):
            print(f"\n{i}. Source: {source}")
            print(f"   Title: {title}")
            print(f"   Content: {content[:150]}...")

cursor.close()
conn.close()
