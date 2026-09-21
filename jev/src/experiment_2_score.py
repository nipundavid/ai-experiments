import os
from typing import TypedDict

from langchain_typesafe import Score, TypeSafeClassifier
from langgraph.graph import END, START, StateGraph


classifier = TypeSafeClassifier(api_key=os.environ["TYPESAFE_API_KEY"])


class State(TypedDict):
    query: str
    complexity: float
    rubric: list[str]


def score_query(state: State):
    response = classifier.invoke(
        {
            "state": state["query"],
            "questions": {
                "complexity": Score(
                    instructions="How complex is this query to answer accurately?",
                    criteria=[
                        "Can be answered directly in one or two sentences.",
                        "Needs a short explanation or a small number of steps.",
                        "Needs substantial reasoning, multiple steps, or external context.",
                    ],
                )
            },
        }
    )

    answer = response.scores["complexity"]
    return {"complexity": answer.score, "rubric": answer.legend}


graph = StateGraph(State)
graph.add_node("score_query", score_query)
graph.add_edge(START, "score_query")
graph.add_edge("score_query", END)
app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"query": "Explain how hybrid search combines keyword and vector search."})
    print(f"Query: {result['query']}")
    print(f"Complexity score: {result['complexity']:.2f}")
    print(f"Rubric: {result['rubric']}")