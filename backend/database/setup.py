from config.database import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, EMBED_DIMENSION
from database.connection import connect_to_postgres

def test_supabase_connection():
    """Test the Supabase database connection"""
    print("\n" + "="*50)
    print("Testing Supabase Connection...")
    print("="*50)
    
    try:
        print(f"\n📡 Connecting to:")
        print(f"   Host: {POSTGRES_HOST}")
        print(f"   Port: {POSTGRES_PORT}")
        print(f"   Database: {POSTGRES_DB}")
        print(f"   User: {POSTGRES_USER}")
        
        conn = connect_to_postgres()
        print("\n✅ Connection successful!")
        
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()
            print(f"\n📊 PostgreSQL version:")
            print(f"   {version[0]}")
            
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                );
            """)
            has_vector = cur.fetchone()[0]
            
            if has_vector:
                print("\n✅ pgvector extension is installed")
            else:
                print("\n⚠️  pgvector extension is NOT installed")
                print("   Run: CREATE EXTENSION vector;")
        
        conn.close()
        print("\n✅ Connection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False

def create_table_if_not_exists(conn):
    """Create embeddings table with proper schema"""
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS my_embeddings (
        id                  SERIAL PRIMARY KEY,
        source_id           TEXT,
        chunk_index         INT,
        text_content        TEXT,
        metadata            JSONB,
        embedding           VECTOR({EMBED_DIMENSION})
    );
    """
    
    create_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_my_embeddings_embedding
        ON my_embeddings
        USING ivfflat (embedding vector_l2_ops)
        WITH (lists = 100);
    """

    with conn.cursor() as cur:
        cur.execute(create_table_sql)
        cur.execute(create_index_sql)
        conn.commit()
    
    print("✅ Table 'my_embeddings' created/verified")