from graph.state import State

def router(state: State):
    """Route to appropriate agent based on message type"""
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return "therapist"
    return "logical"