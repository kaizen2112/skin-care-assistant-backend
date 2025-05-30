from langchain_core.messages import HumanMessage
from chatbot import build_graph

def main():
    graph = build_graph()
    state = {"messages": [], "chat_history": []}  # initialize state correctly

    print("🧑‍⚕️ RAG-powered Skincare Assistant ready! (type exit)\n")
    while True:
        ui = input("You: ")
        if ui.strip().lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        state["messages"].append(HumanMessage(content=ui))
        state = graph.invoke(state)
        print(f"Specialist: {state['messages'][-1].content}\n")

if __name__ == "__main__":
    main()
