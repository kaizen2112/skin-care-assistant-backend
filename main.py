# main.py
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body, Path
from pydantic import BaseModel
import aiosqlite # Import aiosqlite to manage the connection
from contextlib import asynccontextmanager

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Import the build function and saver class, NOT a pre-compiled instance
from chatbot import build_compiled_graph_with_checkpointing, AsyncSqliteSaver

# --- Global variables to hold the graph and checkpointer ---
# They will be 'None' until the lifespan startup event populates them.
compiled_graph = None
checkpointer = None # Add global variable for the checkpointer

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on application startup
    print("[main] Lifespan event: startup")
    print("[main] Setting up resources...")
    
    db_file = "skincare_chat_sessions.sqlite"
    db_conn = await aiosqlite.connect(db_file)
    
    # Make checkpointer global so other functions/endpoints can access it
    global checkpointer 
    checkpointer = AsyncSqliteSaver(conn=db_conn)
    print(f"[main] AsyncSqliteSaver initialized with DB: {db_file}")

    global compiled_graph
    compiled_graph = build_compiled_graph_with_checkpointing(checkpointer)
    print("[main] Compiled graph is ready.")
    
    yield # The application is now ready to run

    # This code runs on application shutdown
    print("[main] Lifespan event: shutdown")
    await db_conn.close()
    print("[main] Database connection closed.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Skincare RAG API with Persistent Sessions",
    description="API for Skincare Assistant. Sessions are persistent via SQLite.",
    version="1.3.1", # Incremented version for logging change
    lifespan=lifespan
)

# --- Pydantic Models ---
class ApiChatMessageInput(BaseModel):
    content: str

class ApiChatMessageOutput(BaseModel):
    content: str
    type: str # "human", "ai"

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: ApiChatMessageInput

class ChatResponse(BaseModel):
    session_id: str
    response_message: ApiChatMessageOutput
    full_chat_log: List[ApiChatMessageOutput]

class ChatHistoryResponse(BaseModel):
    session_id: str
    chat_history: List[ApiChatMessageOutput]
    raw_state: Optional[Dict[str, Any]] = None # Optional: for debugging the full state

# Helper function (remains the same)
def convert_lc_message_to_api_output(message: BaseMessage) -> ApiChatMessageOutput:
    msg_type = "unknown"
    if isinstance(message, HumanMessage):
        msg_type = "human"
    elif isinstance(message, AIMessage):
        msg_type = "ai"
    # Add other types if you log them, e.g., SystemMessage
    return ApiChatMessageOutput(content=str(message.content), type=msg_type)


# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest = Body(...)):
    if compiled_graph is None or checkpointer is None: # Also check checkpointer
        raise HTTPException(status_code=503, detail="Service Unavailable: Core components not initialized.")

    session_id = request.session_id or str(uuid.uuid4())
    print(f"[main] Handling POST /chat for session_id: {session_id}")

    graph_input = {"messages": [HumanMessage(content=request.message.content)]}
    config = {"configurable": {"thread_id": session_id}}

    try:
        result_state = await compiled_graph.ainvoke(graph_input, config=config)
    except Exception as e:
        print(f"[main] Error invoking graph for session_id {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing your request: {str(e)}")

    if not result_state or "messages" not in result_state or not result_state["messages"]:
        # Handle case where graph might not return expected messages
        print(f"[main] Warning: Graph for session {session_id} returned unexpected state: {result_state}")
        # You might want to return a more specific error or an empty log
        ai_response_lc_message = AIMessage(content="Sorry, an issue occurred with processing the conversation.")
        api_chat_log = [convert_lc_message_to_api_output(ai_response_lc_message)]
    else:
        ai_response_lc_message = result_state["messages"][-1]
        api_chat_log = [convert_lc_message_to_api_output(msg) for msg in result_state["messages"]]
    
    return ChatResponse(
        session_id=session_id,
        response_message=convert_lc_message_to_api_output(ai_response_lc_message),
        full_chat_log=api_chat_log
    )

@app.get("/chat_history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str = Path(..., description="The ID of the session to retrieve chat history for.")):
    """
    Retrieves the full chat history for a given session ID.
    """
    if checkpointer is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Checkpointer not initialized.")

    print(f"[main] Handling GET /chat_history for session_id: {session_id}")
    config = {"configurable": {"thread_id": session_id}}

    try:
        # Use the checkpointer's 'aget' method to retrieve the saved state (checkpoint)
        saved_state = await checkpointer.aget(config=config)
        # --- DEBUG LOGGING ADDED ---
        print(f"[main DEBUG] Raw saved_state for session {session_id} from checkpointer.aget(): {saved_state}")
        # --- END DEBUG LOGGING ---
    except Exception as e:
        # This might happen if there's an issue with the DB connection or checkpointer logic
        print(f"[main] Error retrieving state for session_id {session_id} from checkpointer: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

    if saved_state is None:
        print(f"[main DEBUG] No saved_state found for session {session_id}. Returning 404.") # DEBUG LOGGING
        raise HTTPException(status_code=404, detail=f"Chat history not found for session_id: {session_id}")

    # The saved_state is a dictionary representing the LangGraph State.
    # We need to extract the 'messages' list.
    messages_from_state = saved_state.get("messages", [])
    # --- DEBUG LOGGING ADDED ---
    print(f"[main DEBUG] Extracted 'messages_from_state' for session {session_id}: {messages_from_state}")
    if not messages_from_state:
        print(f"[main DEBUG] 'messages_from_state' is empty for session {session_id}.")
    # --- END DEBUG LOGGING ---
    
    if not isinstance(messages_from_state, list):
        print(f"[main] Warning: 'messages' in saved state for session {session_id} is not a list: {messages_from_state}")
        api_chat_log = []
    else:
        api_chat_log = []
        for i, msg_data in enumerate(messages_from_state):
            # --- DEBUG LOGGING ADDED ---
            print(f"[main DEBUG] Processing message {i} from state: {msg_data} (type: {type(msg_data)})")
            # --- END DEBUG LOGGING ---
            if isinstance(msg_data, BaseMessage):
                api_chat_log.append(convert_lc_message_to_api_output(msg_data))
            elif isinstance(msg_data, dict): 
                if "content" in msg_data and "type" in msg_data:
                    if msg_data["type"] == "human":
                        api_chat_log.append(convert_lc_message_to_api_output(HumanMessage(content=msg_data["content"])))
                    elif msg_data["type"] == "ai":
                        api_chat_log.append(convert_lc_message_to_api_output(AIMessage(content=msg_data["content"])))
                    else: 
                        print(f"[main DEBUG] Message {i} is dict with unknown type: {msg_data['type']}") # DEBUG
                        api_chat_log.append(ApiChatMessageOutput(content=str(msg_data.get("content")), type="dict_unknown"))
                else: 
                    print(f"[main DEBUG] Message {i} is dict with unexpected structure: {msg_data}") # DEBUG
                    api_chat_log.append(ApiChatMessageOutput(content=str(msg_data), type="raw_dict"))
            else: 
                print(f"[main DEBUG] Message {i} has unknown format: {msg_data}") # DEBUG
                api_chat_log.append(ApiChatMessageOutput(content=str(msg_data), type="unknown_format"))
        # --- DEBUG LOGGING ADDED ---
        print(f"[main DEBUG] Final 'api_chat_log' for session {session_id}: {api_chat_log}")
        # --- END DEBUG LOGGING ---

    include_raw_state = True # Temporarily set to True for debugging the raw state via API response

    return ChatHistoryResponse(
        session_id=session_id,
        chat_history=api_chat_log,
        raw_state=saved_state if include_raw_state else None
    )