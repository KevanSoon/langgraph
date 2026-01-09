import uuid
from abc import ABC, abstractmethod
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig


class BaseMemoryAgent(ABC):
    """Base class for agents with memory capabilities.

    Extracts shared logic from therapist_agent and logical_agent in supabase.py:
    - Memory retrieval from store
    - Memory storage when user says "remember"
    - Message construction with system prompt + memories
    - LLM invocation and response formatting
    """

    def __init__(self, llm):
        self.llm = llm

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Each agent defines its own personality/system prompt."""
        pass

    async def retrieve_memories(self, store: BaseStore, user_id: str, query: str) -> str:
        """Fetch relevant memories for this user.

        Corresponds to supabase.py lines 209-211 and 242-244:
            memories = await store.asearch(namespace, query=str(last_message.content))
            memory_info = "\\n".join([d.value.get("data", "") for d in memories])
        """
        namespace = ("memories", user_id)
        memories = await store.asearch(namespace, query=query)
        return "\n".join([d.value.get("data", "") for d in memories])

    async def maybe_store_memory(self, store: BaseStore, user_id: str, content: str):
        """Store memory if user asks to remember something.

        Corresponds to supabase.py lines 227-230 and 259-262:
            if "remember" in last_message.content.lower():
                memory_id = str(uuid.uuid4())
                await store.aput(namespace, memory_id, {"data": last_message.content})
        """
        if "remember" in content.lower():
            memory_id = str(uuid.uuid4())
            namespace = ("memories", user_id)
            await store.aput(namespace, memory_id, {"data": content})
            print(f"Stored memory for user {user_id}")

    async def __call__(self, state: dict, config: RunnableConfig, *, store: BaseStore) -> dict:
        """Make the agent callable for LangGraph node compatibility.

        This method replaces the duplicated logic in:
        - therapist_agent (supabase.py lines 203-233)
        - logical_agent (supabase.py lines 235-265)
        """
        last_message = state["messages"][-1]
        user_id = config["configurable"].get("user_id", "default_user")

        # Get memories
        memory_info = await self.retrieve_memories(store, user_id, str(last_message.content))

        # Build prompt with memories injected
        full_prompt = f"""{self.system_prompt}

        User information from previous sessions:
        {memory_info}"""

        messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": last_message.content}
        ]

        # Store memory if requested
        await self.maybe_store_memory(store, user_id, last_message.content)

        # Get response from LLM
        reply = self.llm.invoke(messages)
        return {"messages": [{"role": "assistant", "content": reply.content}]}
