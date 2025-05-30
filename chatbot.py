# #chatbot.py
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain.chains import ConversationalRetrievalChain
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langgraph.graph import StateGraph
# from state import State

# import os
# from dotenv import load_dotenv
# load_dotenv()

# # --- 1. Global Initializations (Load once on app startup) ---
# print("[chatbot] Initializing components...")
# embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # 2. Load vectorstore with embeddings (enable deserialization safety flag)
# vectorstore = FAISS.load_local(
#     "vectorstore", 
#     embedder,
#     allow_dangerous_deserialization=True  # ensure you trust the data source
# )

# # 3. Initialize Groq LLM
# # GROQ_API_KEY = "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"      
# # MODEL_NAME   = "llama3-8b-8192"
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# MODEL_NAME = os.getenv("MODEL_NAME")  # default to llama3-8b-8192 if not set

# llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME)

# # 4. Build Retrieval QA chain
# retrieval_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=vectorstore.as_retriever(),
#     return_source_documents=True
# )

# # 5. System prompt for context
# system_prompt = SystemMessage(content=(
#         "You are a board-certified skincare specialist. "
#     "Use the retrieved product info to give accurate, safe, and context-aware advice, after asking clarifying questions and having their permission before handing out any skin care products.  "
#     "Always provide professional, safe, and tailored advice. "
#     "Before giving recommendations, ask critical clarifying questions "
#     "about the user's skin type, sensitivities, medical conditions, environment, and routine. "
#     "Never assume information and avoid making risky recommendations without full context. "
#     "Your tone should be compassionate and medically responsible."
# ))

# def build_graph():
#     def rag_node(state: State) -> State:
#         # ensure keys
#         state.setdefault("messages", [])
#         state.setdefault("chat_history", [])

#         # inject system prompt on very first turn
#         if not state["chat_history"] and not state["messages"]:
#             state["messages"].append(system_prompt)

#         # user's last message
#         user_msg = state["messages"][-1]

#         # RAG call
#         out = retrieval_chain.invoke({
#             "question": user_msg.content,
#             "chat_history": state["chat_history"]
#         })

#         # append assistant response
#         state["messages"].append(HumanMessage(content=out["answer"]))
#         # update history
#         state["chat_history"] = out["chat_history"]
#         return state

#     g = StateGraph(State)
#     g.add_node("rag_chat", rag_node)
#     g.set_entry_point("rag_chat")
#     g.set_finish_point("rag_chat")
#     return g.compile()

# chatbot.py
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # Added AIMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph

from state import State # Your TypedDict for graph state

load_dotenv()

# --- 1. Global Initializations (Load once on app startup) ---
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
    raise  # Critical error, app should not start

# --- 3. Initialize Groq LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192") # Default if not set

if not GROQ_API_KEY:
    print("[chatbot] GROQ_API_KEY not found in environment variables.")
    raise ValueError("GROQ_API_KEY is not set.")

llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME)
print(f"[chatbot] LLM initialized with model: {MODEL_NAME}")

# --- 4. Build Retrieval QA chain ---
# Note: For more direct persona control, you could use `combine_docs_chain_kwargs`
# with a custom prompt template here. The current approach prepends system message
# content to the first user question.
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True # Set to False if not using them in the final response to save bandwidth
)
print("[chatbot] ConversationalRetrievalChain initialized.")

# --- 5. System prompt content (used in rag_node for the first turn) ---
system_prompt_content = (
    "You are a board-certified skincare specialist. "
    "Use the retrieved product info to give accurate, safe, and context-aware advice, after asking clarifying questions and having their permission before handing out any skin care products. "
    "Always provide professional, safe, and tailored advice. "
    "Before giving recommendations, ask critical clarifying questions "
    "about the user's skin type, sensitivities, medical conditions, environment, and routine. "
    "Never assume information and avoid making risky recommendations without full context. "
    "Your tone should be compassionate and medically responsible."
)
# We don't create a SystemMessage object globally if its content is just used in a string op.
# system_prompt_message = SystemMessage(content=system_prompt_content) # Not strictly needed as an object here anymore

# --- 6. Build Compiled LangGraph ---
def build_compiled_graph():
    def rag_node(state: State) -> State:
        """
        Processes user input, interacts with the RAG chain, and updates state.
        """
        # state["messages"] is the history of all message objects (Human, AI) from LangGraph state
        # state["chat_history"] is the (Human, AI) tuple list for ConversationalRetrievalChain

        # Get the most recent user message from the graph's current message list
        # LangGraph's `add_messages` ensures state["messages"] has the full history up to the user's last input.
        last_message_in_graph_state = state["messages"][-1]
        
        if isinstance(last_message_in_graph_state, HumanMessage):
            user_question = last_message_in_graph_state.content
        elif isinstance(last_message_in_graph_state, dict) and last_message_in_graph_state.get("type") == "human": # For robustness
            user_question = last_message_in_graph_state.get("content", "")
        else:
            # This should ideally not be reached if API layer sends correct HumanMessage
            print(f"[chatbot] rag_node: Last message was not a HumanMessage. Got: {type(last_message_in_graph_state)}")
            # Fallback or error handling
            return { # Return an error message or empty response
                "messages": [AIMessage(content="Sorry, I couldn't understand your last message.")],
                "chat_history": state["chat_history"] # Keep previous history
            }

        question_for_rag = user_question
        
        # Inject system prompt guidance if it's the first RAG turn in the session (empty RAG chat_history)
        if not state["chat_history"]: # This means it's the first call to the RAG chain for this session
            question_for_rag = system_prompt_content + "\n\nHuman: " + user_question
            print(f"[chatbot] rag_node: First turn for session. Prepended system prompt.")

        # RAG call
        # `state["chat_history"]` is expected to be List[Tuple[HumanMessage, AIMessage]] or similar format
        # compatible with `ConversationalRetrievalChain`.
        try:
            out = retrieval_chain.invoke({
                "question": question_for_rag,
                "chat_history": state["chat_history"] # History BEFORE this turn
            })
        except Exception as e:
            print(f"[chatbot] rag_node: Error invoking retrieval_chain: {e}")
            return {
                "messages": [AIMessage(content=f"Sorry, an error occurred while retrieving information: {e}")],
                "chat_history": state["chat_history"]
            }

        ai_response_message = AIMessage(content=out["answer"])
        
        # Return the new AI message for 'messages' field (add_messages will append it).
        # Return the full updated 'chat_history' from the RAG chain for the next turn.
        return {
            "messages": [ai_response_message],
            "chat_history": out["chat_history"] # This is what the chain uses for context
        }

    graph = StateGraph(State)
    graph.add_node("rag_chat", rag_node)
    graph.set_entry_point("rag_chat")
    graph.set_finish_point("rag_chat")
    compiled_graph = graph.compile()
    print("[chatbot] LangGraph compiled.")
    return compiled_graph

# Initialize the compiled graph globally when this module is imported
compiled_graph = build_compiled_graph()
print("[chatbot] Global compiled_graph is ready.")