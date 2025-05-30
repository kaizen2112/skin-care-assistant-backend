#chatbot.py
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from state import State

import os
from dotenv import load_dotenv
load_dotenv()

# 1. Initialize embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load vectorstore with embeddings (enable deserialization safety flag)
vectorstore = FAISS.load_local(
    "vectorstore", 
    embedder,
    allow_dangerous_deserialization=True  # ensure you trust the data source
)

# 3. Initialize Groq LLM
# GROQ_API_KEY = "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"      
# MODEL_NAME   = "llama3-8b-8192"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")  # default to llama3-8b-8192 if not set

llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME)

# 4. Build Retrieval QA chain
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 5. System prompt for context
system_prompt = SystemMessage(content=(
        "You are a board-certified skincare specialist. "
    "Use the retrieved product info to give accurate, safe, and context-aware advice, after asking clarifying questions and having their permission before handing out any skin care products.  "
    "Always provide professional, safe, and tailored advice. "
    "Before giving recommendations, ask critical clarifying questions "
    "about the user's skin type, sensitivities, medical conditions, environment, and routine. "
    "Never assume information and avoid making risky recommendations without full context. "
    "Your tone should be compassionate and medically responsible."
))

def build_graph():
    def rag_node(state: State) -> State:
        # ensure keys
        state.setdefault("messages", [])
        state.setdefault("chat_history", [])

        # inject system prompt on very first turn
        if not state["chat_history"] and not state["messages"]:
            state["messages"].append(system_prompt)

        # user's last message
        user_msg = state["messages"][-1]

        # RAG call
        out = retrieval_chain.invoke({
            "question": user_msg.content,
            "chat_history": state["chat_history"]
        })

        # append assistant response
        state["messages"].append(HumanMessage(content=out["answer"]))
        # update history
        state["chat_history"] = out["chat_history"]
        return state

    g = StateGraph(State)
    g.add_node("rag_chat", rag_node)
    g.set_entry_point("rag_chat")
    g.set_finish_point("rag_chat")
    return g.compile()