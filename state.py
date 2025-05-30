from typing import Annotated, List, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Stores the sequence of messages exchanged (annotated for LangGraph)
    messages: Annotated[List[Any], add_messages]
    # Stores the chat history used for RAG context retention
    chat_history: List[Any]
