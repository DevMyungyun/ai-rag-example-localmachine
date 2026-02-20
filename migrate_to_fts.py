#!/usr/bin/env python3
"""
Migration Script: Add Full-Text Search Support
Upgrades existing database to support PostgreSQL full-text search with tsvector.
"""

import psycopg2
from dotenv import load_dotenv
import os
import sys
from tqdm import tqdm

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


def check_migration_needed(conn):
    """Check if migration is needed."""
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
        print("   Run web_to_vector_db.py first to create the table.")
        cursor.close()
        return False
    
    # Check if content_tsvector column exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_name = 'web_documents'
            AND column_name = 'content_tsvector'
        );
    """)
    
    tsvector_exists = cursor.fetchone()[0]
    cursor.close()
    
    return not tsvector_exists


def migrate_database():
    """Run the migration."""
    print("\n" + "="*70)
    print("Full-Text Search Migration")
    print("="*70)
    
    conn = get_connection()
    
    # Check if migration is needed
    if not check_migration_needed(conn):
        print("\n✓ Migration already applied or not needed.")
        print("  Database schema is up to date.\n")
        conn.close()
        return True
    
    cursor = conn.cursor()
    
    # Get row count
    cursor.execute("SELECT COUNT(*) FROM web_documents;")
    total_rows = cursor.fetchone()[0]
    
    print(f"\nFound {total_rows} existing document chunks")
    print("\nThis migration will:")
    print("  1. Enable pg_trgm extension for fuzzy matching")
    print("  2. Add content_tsvector column for full-text search")
    print("  3. Populate tsvector for all existing rows")
    print("  4. Create GIN indexes for text search")
    print("  5. Create source index for faster lookups")
    print("  6. Optimize IVFFlat index with better lists parameter")
    print("  7. Run ANALYZE to update query planner statistics")
    
    if total_rows > 0:
        print(f"\n⚠️  This will process {total_rows} rows and may take a few minutes.")
    
    confirm = input("\nProceed with migration? (yes/no): ")
    
    if confirm.lower() not in ['yes', 'y']:
        print("Migration cancelled.")
        cursor.close()
        conn.close()
        return False
    
    print("\n" + "="*70)
    print("Starting migration...")
    print("="*70 + "\n")
    
    try:
        # Step 1: Enable pg_trgm extension
        print("1. Enabling pg_trgm extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        conn.commit()
        print("   ✓ pg_trgm extension enabled")
        
        # Step 2: Add content_tsvector column
        print("\n2. Adding content_tsvector column...")
        cursor.execute("""
            ALTER TABLE web_documents 
            ADD COLUMN IF NOT EXISTS content_tsvector tsvector;
        """)
        conn.commit()
        print("   ✓ Column added")
        
        # Step 3: Populate content_tsvector for existing rows
        if total_rows > 0:
            print(f"\n3. Populating tsvector for {total_rows} rows...")
            print("   (This may take a while for large databases)")
            
            # Use batch updates for better performance
            batch_size = 1000
            cursor.execute("SELECT id FROM web_documents ORDER BY id;")
            all_ids = [row[0] for row in cursor.fetchall()]
            
            with tqdm(total=total_rows, unit=' rows') as pbar:
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i+batch_size]
                    cursor.execute("""
                        UPDATE web_documents
                        SET content_tsvector = to_tsvector('english', content)
                        WHERE id = ANY(%s);
                    """, (batch_ids,))
                    conn.commit()
                    pbar.update(len(batch_ids))
            
            print("   ✓ All rows updated with tsvector")
        else:
            print("\n3. No rows to populate (table is empty)")
        
        # Step 4: Create GIN index for tsvector
        print("\n4. Creating GIN index on content_tsvector...")
        print("   (This may take a while...)")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS web_documents_content_tsvector_idx
            ON web_documents
            USING GIN (content_tsvector);
        """)
        conn.commit()
        print("   ✓ Full-text search index created")
        
        # Step 5: Create GIN trigram index
        print("\n5. Creating GIN trigram index...")
        print("   (This may take a while...)")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS web_documents_content_trgm_idx
            ON web_documents
            USING GIN (content gin_trgm_ops);
        """)
        conn.commit()
        print("   ✓ Trigram index created")
        
        # Step 6: Create source index
        print("\n6. Creating source index...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS web_documents_source_idx
            ON web_documents (source);
        """)
        conn.commit()
        print("   ✓ Source index created")
        
        # Step 7: Optimize IVFFlat index
        print("\n7. Optimizing vector index...")
        lists = max(100, total_rows // 1000) if total_rows > 0 else 100
        
        # Drop old index
        cursor.execute("DROP INDEX IF EXISTS web_documents_embedding_idx;")
        
        # Create new index with optimal lists
        cursor.execute(f"""
            CREATE INDEX web_documents_embedding_idx 
            ON web_documents 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists});
        """)
        conn.commit()
        print(f"   ✓ Vector index recreated with lists={lists}")
        
        # Step 8: Run ANALYZE
        print("\n8. Updating query planner statistics...")
        cursor.execute("ANALYZE web_documents;")
        conn.commit()
        print("   ✓ Statistics updated")
        
        # Verification
        print("\n" + "="*70)
        print("Verifying migration...")
        print("="*70 + "\n")
        
        # Check indexes
        cursor.execute("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'web_documents'
            ORDER BY indexname;
        """)
        
        indexes = cursor.fetchall()
        print(f"✓ Created {len(indexes)} indexes:")
        for idx_name, idx_def in indexes:
            print(f"  - {idx_name}")
        
        # Check column
        cursor.execute("""
            SELECT COUNT(*) 
            FROM web_documents 
            WHERE content_tsvector IS NOT NULL;
        """)
        populated = cursor.fetchone()[0]
        print(f"\n✓ {populated}/{total_rows} rows have tsvector populated")
        
        print("\n" + "="*70)
        print("Migration completed successfully!")
        print("="*70)
        print("\nYour database now supports:")
        print("  • Fast full-text search with tsvector")
        print("  • Fuzzy text matching with pg_trgm")
        print("  • Optimized hybrid search (vector + text)")
        print("  • Better query performance with proper indexes")
        print("\nYou can now run queries with improved accuracy:")
        print("  python rag_query.py -q \"your question\"")
        print("="*70 + "\n")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        conn.rollback()
        cursor.close()
        conn.close()
        return False


def rollback_migration():
    """Rollback the migration (remove new features)."""
    print("\n⚠️  WARNING: This will remove full-text search features!")
    print("   The following will be dropped:")
    print("   - content_tsvector column")
    print("   - All new indexes")
    print("   - pg_trgm extension")
    
    confirm = input("\nProceed with rollback? Type 'ROLLBACK' to confirm: ")
    
    if confirm != 'ROLLBACK':
        print("Rollback cancelled.")
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        print("\nRolling back migration...")
        
        # Drop indexes
        print("  Dropping indexes...")
        cursor.execute("DROP INDEX IF EXISTS web_documents_content_tsvector_idx;")
        cursor.execute("DROP INDEX IF EXISTS web_documents_content_trgm_idx;")
        cursor.execute("DROP INDEX IF EXISTS web_documents_source_idx;")
        
        # Drop column
        print("  Dropping content_tsvector column...")
        cursor.execute("ALTER TABLE web_documents DROP COLUMN IF EXISTS content_tsvector;")
        
        # Note: We don't drop pg_trgm extension as it might be used elsewhere
        print("  Note: pg_trgm extension not dropped (may be used by other tables)")
        
        conn.commit()
        print("\n✓ Rollback completed")
        
    except Exception as e:
        print(f"\n❌ Rollback failed: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Migrate database to support full-text search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run migration
  python migrate_to_fts.py

  # Rollback migration
  python migrate_to_fts.py --rollback

  # Check if migration is needed
  python migrate_to_fts.py --check
        """
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Rollback the migration (remove full-text search features)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if migration is needed without applying it'
    )
    
    args = parser.parse_args()
    
    try:
        if args.rollback:
            rollback_migration()
        elif args.check:
            conn = get_connection()
            needed = check_migration_needed(conn)
            conn.close()
            
            if needed:
                print("\n📋 Migration is needed")
                print("   Run: python migrate_to_fts.py")
            else:
                print("\n✓ No migration needed")
                print("   Database schema is up to date")
        else:
            success = migrate_database()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
