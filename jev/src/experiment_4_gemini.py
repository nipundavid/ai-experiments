import os
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_typesafe import Choice, TypeSafeClassifier
from langgraph.graph import END, START, StateGraph


classifier = TypeSafeClassifier(api_key=os.environ["TYPESAFE_API_KEY"])
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    api_key=os.environ["GOOGLE_API_KEY"],
)


class State(TypedDict, total=False):
    query: str
    category: str
    answer: str


def classify_query(state: State):
    response = classifier.invoke(
        {
            "state": state["query"],
            "questions": {
                "category": Choice(
                    instructions="Choose the best route for this query.",
                    criteria={
                        "rag": "The query is about RAG or retrieval.",
                        "coding": "The query is about code.",
                        "general": "The query fits neither category.",
                    },
                )
            },
        }
    )
    return {"category": response.choices["category"].choice}


def ask_gemini(state: State):
    response = llm.invoke(
        f"Answer this user query clearly and concisely:\n\n{state['query']}"
    )
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return {"answer": content}


graph = StateGraph(State)
graph.add_node("classify_query", classify_query)
graph.add_node("ask_gemini", ask_gemini)
graph.add_edge(START, "classify_query")
graph.add_edge("classify_query", "ask_gemini")
graph.add_edge("ask_gemini", END)
app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"query": "Explain why retrieval can improve an LLM answer."})
    print(f"JEv route: {result['category']}")
    print(f"Gemini answer:\n{result['answer']}")