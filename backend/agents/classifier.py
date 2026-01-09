from typing import Literal
from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig


class MessageClassifier(BaseModel):
    """Classification result for routing messages.

    Corresponds to supabase.py lines 159-163.
    """
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )


async def classify_message(state: dict, config: RunnableConfig, *, store: BaseStore, llm) -> dict:
    """Classify user message as emotional or logical.

    Corresponds to classify_messages in supabase.py (lines 169-194).

    Args:
        state: Graph state containing messages
        config: Runtime config with user_id, thread_id
        store: Memory store (required by graph but not used here)
        llm: Language model instance

    Returns:
        Dict with message_type: "emotional" or "logical"
    """
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
- 'logical': if it asks for facts, information, logical analysis, or practical solutions"""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ])
    return {"message_type": result.message_type}



def create_classifier(llm):
    """Factory to create classifier function with LLM bound.

    Usage:
        llm = ChatOllama(model="gpt-oss:120b-cloud")
        classify = create_classifier(llm)
        graph_builder.add_node("classifier", classify)
    """
    async def classifier_node(state: dict, config: RunnableConfig, *, store: BaseStore):
        return await classify_message(state, config, store=store, llm=llm)

    return classifier_node
