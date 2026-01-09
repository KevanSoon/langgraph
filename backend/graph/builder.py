from langgraph.graph import StateGraph, START, END, MessagesState
from graph.state import State
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from agents.therapist import TherapistAgent
from agents.logical import LogicalAgent
from agents.classifier import create_classifier
from langchain_ollama import ChatOllama
from database.connection import create_async_pool
from graph.router import router

async def build_graph_with_memory():
    """Build the graph with Supabase-backed checkpointer and store using AsyncConnectionPool"""
    
    # Create async connection pool
    pool = create_async_pool()
    
    await pool.open()
    
    # Create checkpointer and store from the pool
    checkpointer = AsyncPostgresSaver(pool)
    store = AsyncPostgresStore(pool)
    
    # Setup tables for store and checkpointer
    print("\n🔧 Setting up LangGraph store and checkpointer tables...")
    await checkpointer.setup()
    await store.setup()
    print("✅ Store and checkpointer tables created!\n")
    
    llm = ChatOllama(model="gpt-oss:120b-cloud")

    # Build the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("classifier", create_classifier(llm))
    graph_builder.add_node("therapist", TherapistAgent(llm))
    graph_builder.add_node("logical", LogicalAgent(llm))
    
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