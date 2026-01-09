import asyncio
import sys

# Windows-specific fix for psycopg async compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from cli.chatbot import run_chatbot_with_memory, demo_memory_system
from database.setup import test_supabase_connection


if __name__ == "__main__":
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
            print("  python main.py test      # Test Supabase connection")
            print("  python main.py demo      # Run memory demo")
            print("  python main.py chatbot   # Run interactive chatbot")
    else:
        # Default: run chatbot
        asyncio.run(run_chatbot_with_memory())
