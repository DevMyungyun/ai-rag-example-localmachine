#!/usr/bin/env python3
"""
Manage database: view, clean, or reset data
"""

import psycopg2
from dotenv import load_dotenv
import os
import sys

load_dotenv()


def get_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        database=os.getenv("POSTGRES_DB", "rag_database"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres")
    )


def show_statistics():
    """Show database statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'web_documents'
        );
    """)
    
    if not cursor.fetchone()[0]:
        print("❌ Table 'web_documents' does not exist!")
        return
    
    # Total chunks
    cursor.execute("SELECT COUNT(*) FROM web_documents;")
    total = cursor.fetchone()[0]
    print(f"\n📊 Total chunks: {total}")
    
    if total == 0:
        print("Database is empty.")
        return
    
    # By source
    cursor.execute("""
        SELECT source, COUNT(*) as count 
        FROM web_documents 
        GROUP BY source 
        ORDER BY count DESC;
    """)
    
    print("\n📄 Documents by source:")
    for source, count in cursor.fetchall():
        print(f"  {count:4d} chunks - {source}")
    
    cursor.close()
    conn.close()


def delete_by_source(source_pattern: str):
    """Delete documents by source URL pattern."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Find matching sources
    cursor.execute("""
        SELECT DISTINCT source 
        FROM web_documents 
        WHERE source ILIKE %s;
    """, (f"%{source_pattern}%",))
    
    sources = cursor.fetchall()
    
    if not sources:
        print(f"No sources found matching: {source_pattern}")
        return
    
    print(f"\nFound {len(sources)} matching source(s):")
    for (source,) in sources:
        print(f"  - {source}")
    
    confirm = input(f"\nDelete all data from these {len(sources)} source(s)? (yes/no): ")
    
    if confirm.lower() in ['yes', 'y']:
        cursor.execute("""
            DELETE FROM web_documents 
            WHERE source ILIKE %s;
        """, (f"%{source_pattern}%",))
        
        deleted = cursor.rowcount
        conn.commit()
        print(f"✓ Deleted {deleted} chunks")
    else:
        print("Cancelled")
    
    cursor.close()
    conn.close()


def reset_database():
    """Reset the entire database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM web_documents;")
    total = cursor.fetchone()[0]
    
    print(f"\n⚠️  WARNING: This will delete ALL {total} chunks from the database!")
    confirm = input("Are you sure? Type 'DELETE ALL' to confirm: ")
    
    if confirm == 'DELETE ALL':
        cursor.execute("TRUNCATE TABLE web_documents;")
        conn.commit()
        print("✓ Database reset complete")
    else:
        print("Cancelled")
    
    cursor.close()
    conn.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage vector database')
    parser.add_argument('action', choices=['stats', 'delete', 'reset'], 
                       help='Action to perform')
    parser.add_argument('--source', '-s', type=str,
                       help='Source URL pattern for delete action')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'stats':
            show_statistics()
        elif args.action == 'delete':
            if not args.source:
                print("Error: --source required for delete action")
                parser.print_help()
                sys.exit(1)
            delete_by_source(args.source)
        elif args.action == 'reset':
            reset_database()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
