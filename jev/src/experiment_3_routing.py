import os
from typing import Literal, TypedDict

from langchain_typesafe import Choice, TypeSafeClassifier
from langgraph.graph import END, START, StateGraph


classifier = TypeSafeClassifier(api_key=os.environ["TYPESAFE_API_KEY"])
Category = Literal["rag", "coding", "system_design", "general"]


class State(TypedDict, total=False):
    query: str
    category: Category
    result: str


def classify_query(state: State):
    response = classifier.invoke(
        {
            "state": state["query"],
            "questions": {
                "category": Choice(
                    instructions="Classify the query into exactly one category.",
                    criteria={
                        "rag": "RAG, retrieval, embeddings, or vector databases.",
                        "coding": "Writing, debugging, or understanding code.",
                        "system_design": "Software architecture or system design.",
                        "general": "Anything that does not fit the other categories.",
                    },
                )
            },
        }
    )
    return {"category": response.choices["category"].choice}


def route_query(state: State) -> Category:
    return state["category"]


def make_worker(label: str):
    def worker(state: State):
        return {"result": f"{label} worker selected for: {state['query']}"}

    return worker


graph = StateGraph(State)
graph.add_node("classify_query", classify_query)
for category in ("rag", "coding", "system_design", "general"):
    graph.add_node(category, make_worker(category))

graph.add_edge(START, "classify_query")
graph.add_conditional_edges(
    "classify_query",
    route_query,
    {category: category for category in ("rag", "coding", "system_design", "general")},
)
for category in ("rag", "coding", "system_design", "general"):
    graph.add_edge(category, END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"query": "How should I design a RAG pipeline?"})
    print(f"Category: {result['category']}")
    print(result["result"])