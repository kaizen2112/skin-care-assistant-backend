import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from langgraph.checkpoint.aiosqlite import AioSqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver    


from state import State

load_dotenv()

# --- 1. Global Initializations ---
print("[chatbot] Initializing components...")
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 2. Load vectorstore ---
VECTORSTORE_PATH = "vectorstore"
try:
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embedder,
        allow_dangerous_deserialization=True
    )
    print(f"[chatbot] Vectorstore loaded from {VECTORSTORE_PATH}")
except RuntimeError as e:
    print(f"[chatbot] Error loading vectorstore from {VECTORSTORE_PATH}: {e}")
    print("[chatbot] Please ensure you have run index.py to build and save the vectorstore first.")
    raise

# --- 3. Initialize Groq LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")

if not GROQ_API_KEY:
    print("[chatbot] GROQ_API_KEY not found in environment variables.")
    raise ValueError("GROQ_API_KEY is not set.")

llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME)
print(f"[chatbot] LLM initialized with model: {MODEL_NAME}")

# --- 4. Build Retrieval QA chain ---
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
print("[chatbot] ConversationalRetrievalChain initialized.")

# --- 5. System prompt content ---
system_prompt_content = (
    "You are a board-certified skincare specialist. "
    # "Use the retrieved product info to give accurate, safe, and context-aware advice, after asking clarifying questions and having their permission before handing out any skin care products. "
   " Use the retrieved product info to give accurate, safe, and context-aware advice. suggest the products based on user's skin issue. "
    "Before giving recommendations, ask a few critical clarifying questions "
    "about the user's skin type, sensitivities, medical conditions, environment, and routine. "
    # "Never assume information and avoid making risky recommendations without full context. "
    "Your tone should be compassionate and medically responsible."
)

# --- 2. Define the Graph Building Function (but don't call it here) ---
def build_compiled_graph_with_checkpointing(graph_checkpointer):
    """
    This function takes a fully initialized checkpointer and compiles the graph.
    It will be called by main.py during the application's lifespan startup.
    """
    async def rag_node(state: State) -> State:
        # The logic inside rag_node remains exactly the same as before
        last_message_in_graph_state = state["messages"][-1]
        user_question = last_message_in_graph_state.content
        question_for_rag = user_question
        current_rag_history = state.get("chat_history", [])
        if not current_rag_history:
            question_for_rag = system_prompt_content + "\n\nHuman: " + user_question
        out = await retrieval_chain.ainvoke({
            "question": question_for_rag,
            "chat_history": current_rag_history
        })
        return {
            "messages": [AIMessage(content=out["answer"])],
            "chat_history": out["chat_history"]
        }

    graph = StateGraph(State)
    graph.add_node("rag_chat", rag_node)
    graph.set_entry_point("rag_chat")
    graph.set_finish_point("rag_chat")
    
    # Compile with the provided checkpointer
    compiled_graph_instance = graph.compile(checkpointer=graph_checkpointer)
    print("[chatbot] LangGraph compiled with checkpointer.")
    return compiled_graph_instance