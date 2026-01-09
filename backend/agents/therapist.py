from agents.base import BaseMemoryAgent


class TherapistAgent(BaseMemoryAgent):
    """Emotional/therapeutic response agent.

    Corresponds to therapist_agent in supabase.py (lines 203-233).
    The system prompt is extracted from lines 213-218.
    """

    @property
    def system_prompt(self) -> str:
        return """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
Show empathy, validate their feelings, and help them process their emotions.
Ask thoughtful questions to help them explore their feelings more deeply."""
