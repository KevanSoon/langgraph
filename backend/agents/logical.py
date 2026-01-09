from agents.base import BaseMemoryAgent


class LogicalAgent(BaseMemoryAgent):
    """Logical/factual response agent.

    Corresponds to logical_agent in supabase.py (lines 235-265).
    The system prompt is extracted from lines 245-250.
    """

    @property
    def system_prompt(self) -> str:
        return """You are a purely logical assistant. Focus only on facts and information.
Provide clear, concise answers based on logic and evidence.
Be direct and straightforward in your responses."""
