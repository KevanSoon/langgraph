from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama

#for supabase
import os
import psycopg2
import json

load_dotenv()

#postgres connection info
POSTGRES_HOST = os.getenv("SUPABASE_DB_HOST","localhost")
POSTGRES_PORT = int(os.getenv("SUPABASE_DB_PORT",5432))
POSTGRES_DB = os.getenv("SUPABASE_DB_NAME","postgres")
POSTGRES_USER = os.getenv("SUPABASE_DB_USER","postgres")  # Fixed: removed '='
POSTGRES_PASS = os.getenv("SUPABASE_DB_PASSWORD","")

EMBED_DIMENSION = 1024

def connect_to_postgres():
    """Connect to Supabase PostgreSQL database"""
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASS
    )
    return conn

def test_supabase_connection():
    """Test the Supabase database connection"""
    print("\n" + "="*50)
    print("Testing Supabase Connection...")
    print("="*50)
    
    try:
        # Attempt to connect
        print(f"\n📡 Connecting to:")
        print(f"   Host: {POSTGRES_HOST}")
        print(f"   Port: {POSTGRES_PORT}")
        print(f"   Database: {POSTGRES_DB}")
        print(f"   User: {POSTGRES_USER}")
        
        conn = connect_to_postgres()
        print("\n✅ Connection successful!")
        
        # Test the connection with a simple query
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()
            print(f"\n📊 PostgreSQL version:")
            print(f"   {version[0]}")
            
            # Check if vector extension is available
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
            
            # List existing tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cur.fetchall()
            
            print(f"\n📋 Existing tables in database:")
            if tables:
                for table in tables:
                    print(f"   - {table[0]}")
            else:
                print("   (no tables yet)")
        
        conn.close()
        print("\n✅ Connection test completed successfully!")
        return True
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ Connection failed!")
        print(f"   Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your .env file has the correct credentials")
        print("   2. Verify your Supabase project is running")
        print("   3. Check if your IP is allowed in Supabase settings")
        print("   4. Ensure you're using the direct database URL, not the pooler URL")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def create_table_if_not_exists(conn):
    """Create embeddings table with proper schema"""
    # Fixed: column name 'embeddings' -> 'embedding'
    # Fixed: index name references correct column
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

def chunk_text(text, max_chars=2000):
    """Split text into chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks

def get_embedding(text):
    """Generate embeddings for text (placeholder)"""
    # TODO: Implement with your embedding model
    # Example: return model.encode(text)
    pass

def insert_embedding_row(
        conn, source_id, chunk_idx, text_content, metadata_dict, embedding_vector
):
    """Insert a row into the embeddings table"""
    # Fixed: typos in SQL and cursor
    insert_query = """
        INSERT INTO my_embeddings (source_id, chunk_index, text_content, metadata, embedding)
        VALUES (%s,%s,%s,%s,%s);
    """
    with conn.cursor() as cur:
        cur.execute(
            insert_query,
            (
                source_id,
                chunk_idx,
                text_content,
                json.dumps(metadata_dict),
                embedding_vector                
            )
        )
        conn.commit()


llm = ChatOllama(
    model="gpt-oss:120b-cloud"
)

class MessageClassifier(BaseModel):
    message_type: Literal["emotional","logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

class State(TypedDict): 
    messages: Annotated[list, add_messages]
    message_type: str | None

def classify_messsages(state: State):
    """Classify user message as emotional or logical"""
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either 'emotional' or 'logical'.
            
            Respond ONLY with valid JSON in this exact format:
            {"message_type": "emotional"}
            or
            {"message_type": "logical"}
            
            Rules:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            
            Do not include any other text, just the JSON object.
            """
        },
        {
            "role": "user",
            "content": last_message.content
        }  
    ])
    return {"message_type": result.message_type}

def router(state: State):
    """Route to appropriate agent based on message type"""
    message_type = state.get("message_type","logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}

def therapist_agent(state: State):
    """Handle emotional/therapeutic responses"""
    last_message = state["messages"][-1]
    messages = [
        {"role": "system", 
         "content": """
            You are a compassionate therapist. Focus on the emotional aspects of the user's message.
            Show empathy, validate their feelings, and help them process their emotions.
            Ask thoughtful questions to help them explore their feelings more deeply.
            Avoid giving logical solutions unless explicitly asked.
            """
         },
         {
             "role": "user",
             "content": last_message.content
         }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def logical_agent(state: State):
    """Handle logical/factual responses"""
    last_message = state["messages"][-1]
    messages = [
        {"role": "system", 
        "content": """
            You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses.
            """
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

# Building the graph
graph_builder = StateGraph(State)
graph_builder.add_node("classifier",classify_messsages)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist",therapist_agent)
graph_builder.add_node("logical",logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier","router")
graph_builder.add_conditional_edges(
     "router",
     lambda state: state.get("next"),
     {"therapist": "therapist","logical":"logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)
graph = graph_builder.compile()

def run_chatbot():
    """Run the interactive chatbot"""
    state = {"messages": [], "message_type": None}

    while True:
         user_input = input("Message: ")
         if user_input.lower() in ['exit', 'quit']:
            print("Bye")
            break
         
         state["messages"] = state.get("messages",[]) + [
             {"role": "user", "content": user_input}
         ]

         state = graph.invoke(state)

         if state.get("messages") and len(state["messages"]) > 0:
             last_message = state["messages"][-1]
             print(f"Assistant: {last_message.content}")

def initialize_database():
    """Initialize database: test connection and create tables"""
    print("\n" + "="*50)
    print("Initializing Database...")
    print("="*50)
    
    # Test connection
    if not test_supabase_connection():
        print("\n❌ Please fix the database connection first.")
        return False
    
    # Create tables
    try:
        print("\n📝 Creating tables...")
        conn = connect_to_postgres()
        create_table_if_not_exists(conn)
        conn.close()
        print("✅ Database initialization complete!\n")
        return True
    except Exception as e:
        print(f"\n❌ Error creating tables: {e}")
        return False

def setup_database_with_sample_data():
    """Setup database and insert sample embeddings (for testing)"""
    if not initialize_database():
        return
    
    print("\n" + "="*50)
    print("Inserting Sample Data...")
    print("="*50)
    
    try:
        conn = connect_to_postgres()
        
        # Sample data
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science."
        ]
        
        for idx, text in enumerate(sample_texts):
            # Create a dummy embedding (you'll replace this with real embeddings)
            dummy_embedding = [0.1] * EMBED_DIMENSION  # Replace with actual embeddings
            
            metadata = {
                "source": "sample_document",
                "created_at": "2024-01-01",
                "category": "test"
            }
            
            insert_embedding_row(
                conn=conn,
                source_id=f"doc_{idx}",
                chunk_idx=idx,
                text_content=text,
                metadata_dict=metadata,
                embedding_vector=dummy_embedding
            )
            print(f"✅ Inserted sample row {idx + 1}")
        
        # Verify insertion
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM my_embeddings;")
            count = cur.fetchone()[0]
            print(f"\n📊 Total rows in my_embeddings: {count}")
            
            # Show sample data
            cur.execute("""
                SELECT id, source_id, chunk_index, 
                       LEFT(text_content, 50) as text_preview
                FROM my_embeddings 
                LIMIT 3;
            """)
            rows = cur.fetchall()
            print("\n📋 Sample data:")
            for row in rows:
                print(f"   ID: {row[0]}, Source: {row[1]}, Chunk: {row[2]}")
                print(f"   Text: {row[3]}...\n")
        
        conn.close()
        print("✅ Sample data inserted successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Error inserting sample data: {e}")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            # Just test the connection
            test_supabase_connection()
            
        elif command == "init":
            # Initialize database (create tables)
            initialize_database()
            
        elif command == "setup":
            # Full setup with sample data
            setup_database_with_sample_data()
            
        elif command == "chatbot":
            # Run chatbot
            if initialize_database():
                print("\n" + "="*50)
                print("Starting chatbot...")
                print("="*50 + "\n")
                run_chatbot()
            else:
                print("\n❌ Please fix the database first.")
        else:
            print("Usage:")
            print("  python script.py test      # Test connection only")
            print("  python script.py init      # Initialize database (create tables)")
            print("  python script.py setup     # Full setup with sample data")
            print("  python script.py chatbot   # Run chatbot with DB init")
    else:
        # Default: just run chatbot with initialization
        if initialize_database():
            print("\n" + "="*50)
            print("Starting chatbot...")
            print("="*50 + "\n")
            run_chatbot()
        else:
            print("\n❌ Please fix the database connection before running the chatbot.")