from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from psycopg_pool import AsyncConnectionPool

import os
import psycopg2
import json
import uuid
import asyncio
import sys

# Fix for Windows ProactorEventLoop issue with psycopg
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import async postgres modules - will fail if psycopg[binary] not installed
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from langgraph.store.base import BaseStore
    from langchain_core.runnables import RunnableConfig
    ASYNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Async PostgreSQL features not available: {e}")
    print("\n📦 To fix this, install the required package:")
    print("   pip install 'psycopg[binary]'")
    print("   or")
    print("   pip install psycopg-binary\n")
    ASYNC_AVAILABLE = False

load_dotenv()

# Postgres connection info
POSTGRES_HOST = os.getenv("SUPABASE_DB_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("SUPABASE_DB_PORT", 5432))
POSTGRES_DB = os.getenv("SUPABASE_DB_NAME", "postgres")
POSTGRES_USER = os.getenv("SUPABASE_DB_USER", "postgres")
POSTGRES_PASS = os.getenv("SUPABASE_DB_PASSWORD", "")

EMBED_DIMENSION = 1024


# Connection string for AsyncConnectionPool
CONNECTION_STRING = (
    f"postgres://{POSTGRES_USER}:{POSTGRES_PASS}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    f"?sslmode=require"
)

print(f"📡 Using connection: postgres://{POSTGRES_USER}:****@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")

def create_async_pool():
    """Create AsyncConnectionPool with proper settings"""
    return AsyncConnectionPool(
        conninfo=CONNECTION_STRING,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": None,
        }
    )

# ============================================================================
# SYNC CONNECTION FUNCTIONS (for testing and setup)
# ============================================================================

def connect_to_postgres():
    """Connect to Supabase PostgreSQL database (sync)"""
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

# ============================================================================
# ASYNC GRAPH WITH STORE & CHECKPOINTER
# ============================================================================

llm = ChatOllama(model="gpt-oss:120b-cloud")

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

class State(TypedDict): 
    messages: Annotated[list, add_messages]
    message_type: str | None

async def classify_messages(state: State, config: RunnableConfig, *, store: BaseStore):
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
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return "therapist"
    return "logical"

async def therapist_agent(state: State, config: RunnableConfig, *, store: BaseStore):
    """Handle emotional/therapeutic responses with memory"""
    last_message = state["messages"][-1]
    user_id = config["configurable"].get("user_id", "default_user")
    namespace = ("memories", user_id)
    
    # Retrieve relevant memories
    memories = await store.asearch(namespace, query=str(last_message.content))
    memory_info = "\n".join([d.value.get("data", "") for d in memories])
    
    system_content = f"""You are a compassionate therapist. Focus on the emotional aspects of the user's message.
    Show empathy, validate their feelings, and help them process their emotions.
    Ask thoughtful questions to help them explore their feelings more deeply.
    
    User information from previous sessions:
    {memory_info}
    """
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": last_message.content}
    ]
    
    # Store new memories if user asks to remember
    if "remember" in last_message.content.lower():
        memory_id = str(uuid.uuid4())
        await store.aput(namespace, memory_id, {"data": last_message.content})
        print(f"💾 Stored memory for user {user_id}")
    
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

async def logical_agent(state: State, config: RunnableConfig, *, store: BaseStore):
    """Handle logical/factual responses with memory"""
    last_message = state["messages"][-1]
    user_id = config["configurable"].get("user_id", "default_user")
    namespace = ("memories", user_id)
    
    # Retrieve relevant memories
    memories = await store.asearch(namespace, query=str(last_message.content))
    memory_info = "\n".join([d.value.get("data", "") for d in memories])
    
    system_content = f"""You are a purely logical assistant. Focus only on facts and information.
    Provide clear, concise answers based on logic and evidence.
    Be direct and straightforward in your responses.
    
    User information from previous sessions:
    {memory_info}
    """
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": last_message.content}
    ]
    
    # Store new memories if user asks to remember
    if "remember" in last_message.content.lower():
        memory_id = str(uuid.uuid4())
        await store.aput(namespace, memory_id, {"data": last_message.content})
        print(f"💾 Stored memory for user {user_id}")
    
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

# ============================================================================
# BUILD GRAPH WITH CHECKPOINTER & STORE
# ============================================================================

async def build_graph_with_memory():
    """Build the graph with Supabase-backed checkpointer and store using AsyncConnectionPool"""
    
    if not ASYNC_AVAILABLE:
        print("\n❌ Async PostgreSQL features not available!")
        print("Please install: pip install 'psycopg[binary]'")
        return None, None, None
    
    # Create async connection pool
    pool = AsyncConnectionPool(
        conninfo=CONNECTION_STRING,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": None,
        }
    )
    
    await pool.open()
    
    # Create checkpointer and store from the pool
    checkpointer = AsyncPostgresSaver(pool)
    store = AsyncPostgresStore(pool)
    
    # Setup tables for store and checkpointer
    print("\n🔧 Setting up LangGraph store and checkpointer tables...")
    await checkpointer.setup()
    await store.setup()
    print("✅ Store and checkpointer tables created!\n")
    
    # Build the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("classifier", classify_messages)
    graph_builder.add_node("therapist", therapist_agent)
    graph_builder.add_node("logical", logical_agent)
    
    graph_builder.add_edge(START, "classifier")
    graph_builder.add_conditional_edges(
        "classifier",
        router,
        {"therapist": "therapist", "logical": "logical"}
    )
    graph_builder.add_edge("therapist", END)
    graph_builder.add_edge("logical", END)
    
    # Compile with store and checkpointer
    graph = graph_builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
    
    return graph, store, checkpointer

# ============================================================================
# RUN CHATBOT WITH MEMORY
# ============================================================================

async def run_chatbot_with_memory():
    """Run chatbot with persistent memory using Supabase with AsyncConnectionPool"""
    
    if not ASYNC_AVAILABLE:
        print("\n❌ Async PostgreSQL features not available!")
        print("Please install: pip install 'psycopg[binary]'")
        return
    
    # Create async connection pool
    async with AsyncConnectionPool(
        conninfo=CONNECTION_STRING,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": None,
        }
    ) as pool:
        
        # Create checkpointer and store from the pool
        checkpointer = AsyncPostgresSaver(pool)
        store = AsyncPostgresStore(pool)
        
        # Setup tables
        print("\n🔧 Setting up LangGraph store and checkpointer tables...")
        await checkpointer.setup()
        await store.setup()
        print("✅ Store and checkpointer tables created!\n")
        
        # Build graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("classifier", classify_messages)
        graph_builder.add_node("therapist", therapist_agent)
        graph_builder.add_node("logical", logical_agent)
        
        graph_builder.add_edge(START, "classifier")
        graph_builder.add_conditional_edges(
            "classifier",
            router,
            {"therapist": "therapist", "logical": "logical"}
        )
        graph_builder.add_edge("therapist", END)
        graph_builder.add_edge("logical", END)
        
        # Compile graph with checkpointer and store
        graph = graph_builder.compile(
            checkpointer=checkpointer,
            store=store,
        )
        
        print("\n" + "="*50)
        print("Chatbot with Memory (Supabase-backed)")
        print("="*50)
        print("Commands:")
        print("  - Type 'exit' to quit")
        print("  - Say 'remember: <something>' to store a memory")
        print("  - Type 'user: <id>' to switch users")
        print("  - Type 'thread: <id>' to switch threads")
        print("="*50 + "\n")
        
        user_id = "user_1"
        thread_id = "thread_1"
        
        while True:
            user_input = input(f"[User {user_id}] Message: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            # Allow user to change thread or user ID
            if user_input.startswith("thread:"):
                thread_id = user_input.split(":", 1)[1].strip()
                print(f"📝 Switched to thread: {thread_id}")
                continue
            
            if user_input.startswith("user:"):
                user_id = user_input.split(":", 1)[1].strip()
                print(f"👤 Switched to user: {user_id}")
                continue
            
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                }
            }
            
            # Stream response
            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode="values",
            ):
                if chunk.get("messages"):
                    last_msg = chunk["messages"][-1]
                    if hasattr(last_msg, 'content') and last_msg.type == 'ai':
                        print(f"\n🤖 Assistant: {last_msg.content}\n")

# ============================================================================
# DEMO FUNCTION
# ============================================================================

async def demo_memory_system():
    """Demo the memory system with multiple conversations using AsyncConnectionPool"""
    
    if not ASYNC_AVAILABLE:
        print("\n❌ Async PostgreSQL features not available!")
        print("Please install: pip install 'psycopg[binary]'")
        return
    
    # Create async connection pool
    async with AsyncConnectionPool(
        conninfo=CONNECTION_STRING,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": None,
        }
    ) as pool:
        
        # Create checkpointer and store from the pool
        checkpointer = AsyncPostgresSaver(pool)
        store = AsyncPostgresStore(pool)
        
        # Setup tables
        print("\n🔧 Setting up tables...")
        await checkpointer.setup()
        await store.setup()
        print("✅ Tables ready!\n")
        
        # Build graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("classifier", classify_messages)
        graph_builder.add_node("therapist", therapist_agent)
        graph_builder.add_node("logical", logical_agent)
        
        graph_builder.add_edge(START, "classifier")
        graph_builder.add_conditional_edges(
            "classifier",
            router,
            {"therapist": "therapist", "logical": "logical"}
        )
        graph_builder.add_edge("therapist", END)
        graph_builder.add_edge("logical", END)
        
        graph = graph_builder.compile(
            checkpointer=checkpointer,
            store=store,
        )
        
        print("\n" + "="*50)
        print("Memory System Demo")
        print("="*50)
        
        # First conversation - store memory
        print("\n📝 Conversation 1: Storing information...")
        config1 = {"configurable": {"thread_id": "1", "user_id": "alice"}}
        
        async for chunk in graph.astream(
            {"messages": [{"role": "user", "content": "Hi! Remember: my name is Alice and I love Python"}]},
            config1,
            stream_mode="values",
        ):
            if chunk.get("messages"):
                last_msg = chunk["messages"][-1]
                if hasattr(last_msg, 'content') and last_msg.type == 'ai':
                    print(f"🤖 {last_msg.content}")
        
        # Second conversation - recall memory
        print("\n\n📝 Conversation 2: Recalling information...")
        config2 = {"configurable": {"thread_id": "2", "user_id": "alice"}}
        
        async for chunk in graph.astream(
            {"messages": [{"role": "user", "content": "What do you know about me?"}]},
            config2,
            stream_mode="values",
        ):
            if chunk.get("messages"):
                last_msg = chunk["messages"][-1]
                if hasattr(last_msg, 'content') and last_msg.type == 'ai':
                    print(f"🤖 {last_msg.content}")
        
        print("\n✅ Demo complete!")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            test_supabase_connection()
            
        elif command == "demo":
            asyncio.run(demo_memory_system())
            
        elif command == "chatbot":
            asyncio.run(run_chatbot_with_memory())
            
        else:
            print("Usage:")
            print("  python script.py test      # Test Supabase connection")
            print("  python script.py demo      # Run memory demo")
            print("  python script.py chatbot   # Run interactive chatbot")
    else:
        # Default: run chatbot
        asyncio.run(run_chatbot_with_memory())