# # main.py
# from langchain_core.messages import HumanMessage
# from chatbot import build_graph

# def main():
#     graph = build_graph()
#     state = {"messages": [], "chat_history": []}  # initialize state correctly

#     print("🧑‍⚕️ RAG-powered Skincare Assistant ready! (type exit)\n")
#     while True:
#         ui = input("You: ")
#         if ui.strip().lower() in {"exit", "quit"}:
#             print("👋 Goodbye!")
#             break

#         state["messages"].append(HumanMessage(content=ui))
#         state = graph.invoke(state)
#         print(f"Specialist: {state['messages'][-1].content}\n")

# if __name__ == "__main__":
#     main()

# main.py
import uuid
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

# LangChain message types
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # SystemMessage if you plan to log it

# Import the globally initialized compiled_graph from chatbot.py
# This also runs all the global initializations in chatbot.py (LLM, vectorstore etc.)
try:
    from chatbot import compiled_graph
except Exception as e:
    # Catch errors from chatbot.py initialization (e.g., missing .env, vectorstore)
    print(f"[main] Critical error during chatbot module import: {e}")
    print("[main] The application cannot start. Please check configurations and run index.py if needed.")
    exit(1) # Exit if core components fail to load


app = FastAPI(
    title="Skincare RAG API",
    description="API for interacting with the RAG-powered Skincare Assistant. Manage sessions using session_id.",
    version="1.0.0"
)

# --- Session Management ---
# WARNING: In-memory storage. Resets on server restart. For production, use Redis, a DB, etc.
user_sessions: Dict[str, Dict[str, Any]] = {}
# Structure: session_id -> {"messages": List[BaseMessage], "chat_history": List[Tuple[BaseMessage, BaseMessage]]}

# --- Pydantic Models for API ---
class ApiChatMessageInput(BaseModel):
    content: str
    # type: str = Field("human", const=True) # Ensuring it's always human from input

class ApiChatMessageOutput(BaseModel):
    content: str
    type: str # "human", "ai", "system"

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: ApiChatMessageInput

class ChatResponse(BaseModel):
    session_id: str
    response_message: ApiChatMessageOutput
    full_chat_log: List[ApiChatMessageOutput]
    # Optional: include source documents if your retrieval_chain returns them and you want to expose them
    # source_documents: Optional[List[Dict]] = None

# Helper to convert LangChain BaseMessage to Pydantic model for API response
def convert_lc_message_to_api_output(message: BaseMessage) -> ApiChatMessageOutput:
    msg_type = "unknown"
    if isinstance(message, HumanMessage):
        msg_type = "human"
    elif isinstance(message, AIMessage):
        msg_type = "ai"
    # Add SystemMessage if you intend to include them in full_chat_log
    # elif isinstance(message, SystemMessage):
    #     msg_type = "system"
    return ApiChatMessageOutput(content=str(message.content), type=msg_type)

@app.on_event("startup")
async def startup_event():
    # This event is triggered when FastAPI starts up.
    # The import of `compiled_graph` from `chatbot` already initializes components.
    print("[main] FastAPI startup: Skincare Assistant API is ready.")
    if compiled_graph:
        print("[main] Compiled LangGraph loaded successfully.")
    else:
        print("[main] ERROR: Compiled LangGraph is not available. The application might not work.")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest = Body(...)):
    """
    Handles a chat message from the user.
    Manages conversation state using a session_id.
    """
    session_id = request.session_id
    
    if not session_id or session_id not in user_sessions:
        session_id = str(uuid.uuid4())
        # Initialize state for a new session
        user_sessions[session_id] = {
            "messages": [],       # For LangGraph's `State.messages` (full log of HumanMessage, AIMessage)
            "chat_history": []    # For `ConversationalRetrievalChain` (list of (Human, AI) message tuples)
        }
        print(f"[main] New session started: {session_id}")

    current_session_data = user_sessions[session_id]
    
    # Append new user message to the session's "messages" log (for LangGraph)
    user_lc_message = HumanMessage(content=request.message.content)
    current_session_data["messages"].append(user_lc_message)

    # Prepare state for the LangGraph graph invocation
    # The graph's `State` TypedDict expects `messages` and `chat_history`.
    # `add_messages` in `State.messages` means the graph appends to this list.
    graph_input_state = {
        "messages": [user_lc_message], # Pass only the new message due to `add_messages` behavior
                                       # LangGraph will combine it with checkpointed messages.
        "chat_history": list(current_session_data["chat_history"]) # Pass current RAG history
    }
    
    # Configuration for invoking the graph, ensuring it uses the correct checkpoint for this session
    # LangGraph handles checkpointing internally if a checkpointer is configured.
    # For a simple in-memory approach without explicit checkpointer, we manage the full state.
    # If `add_messages` is used, graph needs the running list of messages.
    # Let's pass the full current list of messages, and rag_node should expect this via state['messages']
    # and the AIMessage it returns will be appended by add_messages.
    
    # Correction based on `add_messages` and LangGraph's handling:
    # The graph state is maintained by LangGraph. We provide new inputs to the existing state.
    # The `compiled_graph.invoke` call takes the *current inputs* and a *configuration*
    # that tells it which conversation (checkpoint) to use.
    # For our manual session management, we're effectively managing the checkpoint data ourselves.

    # Let's ensure the `graph_input_state` aligns with what `rag_node` and `State` expect.
    # `State.messages` uses `add_messages`.
    # When invoking graph, if we pass `{"messages": [new_human_message]}`,
    # LangGraph appends this to the existing messages in the checkpoint for that session.
    # The `rag_node` then receives the full list in `state["messages"]`.
    
    # The state for the graph will be the current accumulated state for the session.
    # LangGraph's StateGraph.compile() returns a `CompiledGraph`.
    # Invoking it with the full state dictionary for that "thread" (session) should work.
    
    # Let's try passing the full current state, graph will use it.
    full_graph_input_state_for_session = {
        "messages": list(current_session_data["messages"]), # Full log for this session
        "chat_history": list(current_session_data["chat_history"]) # Current RAG history
    }

    try:
        # Invoke the graph. It operates on the state.
        # The result_state will contain the complete updated state for this session.
        result_state = compiled_graph.invoke(full_graph_input_state_for_session)
        
    except Exception as e:
        print(f"[main] Error invoking graph for session {session_id}: {e}")
        # Consider logging the `full_graph_input_state_for_session` for debugging
        raise HTTPException(status_code=500, detail=f"Error processing your request: {str(e)}")

    # Update our session data with the full state returned by the graph
    user_sessions[session_id]["messages"] = result_state["messages"]
    user_sessions[session_id]["chat_history"] = result_state["chat_history"]

    # The last message in the `result_state["messages"]` should be the AI's response
    ai_response_lc_message = result_state["messages"][-1] if result_state["messages"] else AIMessage(content="No response generated.")
    
    # Prepare API response
    api_chat_log = [convert_lc_message_to_api_output(msg) for msg in result_state["messages"]]
    
    return ChatResponse(
        session_id=session_id,
        response_message=convert_lc_message_to_api_output(ai_response_lc_message),
        full_chat_log=api_chat_log
        # source_documents=result_state.get("source_documents") # If you add this to graph output
    )

# To run (ensure .env is in the same directory or vars are set):
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# Example request using curl:
# curl -X POST "http://localhost:8000/chat" \
# -H "Content-Type: application/json" \
# -d '{
# "session_id": "my-test-session-123",
# "message": {
# "content": "Hi, what are some good moisturizers for dry skin?"
# }
# }'
#
# For a new session (omit session_id or send a new one):
# curl -X POST "http://localhost:8000/chat" \
# -H "Content-Type: application/json" \
# -d '{
# "message": {
# "content": "Hello there!"
# }
# }'