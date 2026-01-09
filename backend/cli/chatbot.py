from graph.builder import build_graph_with_memory


async def run_chatbot_with_memory():
    """Run chatbot with persistent memory using Supabase with AsyncConnectionPool"""

    # Build graph using the shared builder
    graph, _, _ = await build_graph_with_memory()

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
            print("Goodbye!")
            break

        # Allow user to change thread or user ID
        if user_input.startswith("thread:"):
            thread_id = user_input.split(":", 1)[1].strip()
            print(f"Switched to thread: {thread_id}")
            continue

        if user_input.startswith("user:"):
            user_id = user_input.split(":", 1)[1].strip()
            print(f"Switched to user: {user_id}")
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
                    print(f"\nAssistant: {last_msg.content}\n")


# ============================================================================
# DEMO FUNCTION
# ============================================================================

async def demo_memory_system():
    """Demo the memory system with multiple conversations using AsyncConnectionPool"""

    # Build graph using the shared builder
    graph, _, _ = await build_graph_with_memory()

    print("\n" + "="*50)
    print("Memory System Demo")
    print("="*50)

    # First conversation - store memory
    print("\nConversation 1: Storing information...")
    config1 = {"configurable": {"thread_id": "1", "user_id": "alice"}}

    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "Hi! Remember: my name is Alice and I love Python"}]},
        config1,
        stream_mode="values",
    ):
        if chunk.get("messages"):
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, 'content') and last_msg.type == 'ai':
                print(f"Assistant: {last_msg.content}")

    # Second conversation - recall memory
    print("\n\nConversation 2: Recalling information...")
    config2 = {"configurable": {"thread_id": "2", "user_id": "alice"}}

    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "What do you know about me?"}]},
        config2,
        stream_mode="values",
    ):
        if chunk.get("messages"):
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, 'content') and last_msg.type == 'ai':
                print(f"Assistant: {last_msg.content}")

    print("\nDemo complete!")
