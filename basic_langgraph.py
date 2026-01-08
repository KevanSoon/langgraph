from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool

# 1️⃣ Define a tool (optional)
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together and return the result."""
    return a + b
tools = [add]

# 2️⃣ Initialize your Ollama model
llm = ChatOllama(
    model="gpt-oss:120b-cloud"  # e.g., "gpt-oss:120b-cloud"
)

# 3️⃣ Create the ReAct agent
agent = create_react_agent(
    model=llm,
    tools=tools,
)

# 4️⃣ Invoke the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the most expensive country to live in?"}]}
)

# 5️⃣ Inspect the messages
messages = response.get("messages", [])
for msg in messages:
    print(f"{msg['role'] if 'role' in msg else 'tool'}: {msg.content}")
