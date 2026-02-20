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
    
    # Check if tsvector column exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_name = 'web_documents'
            AND column_name = 'content_tsvector'
        );
    """)
    has_fts = cursor.fetchone()[0]
    
    if has_fts:
        cursor.execute("SELECT COUNT(*) FROM web_documents WHERE content_tsvector IS NOT NULL;")
        fts_count = cursor.fetchone()[0]
        print(f"   Full-text search ready: {fts_count}/{total} chunks ({100*fts_count//total if total > 0 else 0}%)")
        if fts_count < total:
            print("   ⚠️  Run migration: python migrate_to_fts.py")
    else:
        print("   ⚠️  Full-text search not available (run: python migrate_to_fts.py)")
    
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


def show_index_stats():
    """Show index usage statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    print("\n📊 Index Usage Statistics\n")
    print("=" * 80)
    
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
    
    # Get all indexes and their stats
    cursor.execute("""
        SELECT 
            i.indexname,
            i.indexdef,
            pg_size_pretty(pg_relation_size(quote_ident(i.schemaname) || '.' || quote_ident(i.indexname))) as size,
            s.idx_scan as scans,
            s.idx_tup_read as tuples_read,
            s.idx_tup_fetch as tuples_fetched
        FROM pg_indexes i
        LEFT JOIN pg_stat_user_indexes s ON i.indexname = s.indexrelname AND i.tablename = s.relname
        WHERE i.tablename = 'web_documents'
        ORDER BY i.indexname;
    """)
    
    indexes = cursor.fetchall()
    
    if not indexes:
        print("No indexes found.")
        return
    
    for idx_name, idx_def, size, scans, tuples_read, tuples_fetched in indexes:
        print(f"\n📇 {idx_name}")
        print(f"   Size: {size}")
        print(f"   Scans: {scans or 0}")
        if scans:
            print(f"   Tuples read: {tuples_read}")
            print(f"   Tuples fetched: {tuples_fetched}")
        print(f"   Definition: {idx_def[:100]}..." if len(idx_def) > 100 else f"   Definition: {idx_def}")
    
    # Overall table stats
    cursor.execute("""
        SELECT 
            pg_size_pretty(pg_total_relation_size('web_documents')) as total_size,
            pg_size_pretty(pg_relation_size('web_documents')) as table_size,
            pg_size_pretty(pg_total_relation_size('web_documents') - pg_relation_size('web_documents')) as indexes_size
    """)
    
    total_size, table_size, indexes_size = cursor.fetchone()
    
    print("\n" + "=" * 80)
    print(f"\n📦 Storage Summary")
    print(f"   Table size: {table_size}")
    print(f"   Indexes size: {indexes_size}")
    print(f"   Total size: {total_size}")
    print("\n" + "=" * 80 + "\n")
    
    cursor.close()
    conn.close()


def reindex_vector():
    """Rebuild vector index with optimal parameters."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get row count
    cursor.execute("SELECT COUNT(*) FROM web_documents;")
    total_rows = cursor.fetchone()[0]
    
    if total_rows == 0:
        print("Database is empty. No need to reindex.")
        return
    
    # Calculate optimal lists parameter
    lists = max(100, total_rows // 1000)
    
    print(f"\n🔧 Reindexing vector index for {total_rows} documents")
    print(f"   Using lists parameter: {lists}")
    print(f"\n⚠️  This may take several minutes for large databases...\n")
    
    confirm = input("Proceed with reindexing? (yes/no): ")
    
    if confirm.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    try:
        # Drop old index
        print("   Dropping old index...")
        cursor.execute("DROP INDEX IF EXISTS web_documents_embedding_idx;")
        conn.commit()
        
        # Create new index
        print("   Creating new index (this may take a while)...")
        cursor.execute(f"""
            CREATE INDEX web_documents_embedding_idx 
            ON web_documents 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists});
        """)
        conn.commit()
        
        print(f"\n✓ Vector index rebuilt successfully with lists={lists}")
        print("\n💡 Run ANALYZE to update query planner statistics:")
        print("   python manage_db.py analyze\n")
        
    except Exception as e:
        print(f"\n❌ Reindexing failed: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def run_analyze():
    """Run ANALYZE to update query planner statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    print("\n📊 Running ANALYZE on web_documents table...")
    print("   This updates query planner statistics for better performance.\n")
    
    try:
        cursor.execute("ANALYZE web_documents;")
        conn.commit()
        print("✓ ANALYZE completed successfully")
        print("\n💡 Query planner now has up-to-date statistics.\n")
    except Exception as e:
        print(f"\n❌ ANALYZE failed: {e}\n")
    finally:
        cursor.close()
        conn.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage vector database')
    parser.add_argument(
        'action', 
        choices=['stats', 'delete', 'reset', 'index-stats', 'reindex', 'analyze'], 
        help='Action to perform'
    )
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
        elif args.action == 'index-stats':
            show_index_stats()
        elif args.action == 'reindex':
            reindex_vector()
        elif args.action == 'analyze':
            run_analyze()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
