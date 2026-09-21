import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_typesafe import Choice, Noul, TypeSafeClassifier


# ============================================================
# Jev
# ============================================================

classifier = TypeSafeClassifier(
    api_key=os.environ["TYPESAFE_API_KEY"]
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class State(TypedDict):
    query: str

    # Decisions made by Jev.
    is_simple: bool
    category: str


# ============================================================
# Jev node
# ============================================================

def analyze_query(state: State):

    query = state["query"]

    response = classifier.invoke({
        "state": query,

        "questions": {

            # ------------------------------------------------
            # NOUL: binary decision represented by a probability.
            # ------------------------------------------------
            # Binary decision.
            #
            # Think:
            #
            #     YES / NO
            #
            # Useful for guardrails, routing, filtering,
            # escalation decisions, etc.
            # ------------------------------------------------

            "is_simple": Noul(
                instructions=(
                    "Is this a simple question that can be "
                    "answered directly without retrieval, "
                    "multiple reasoning steps, or external tools?"
                )
            ),


            # ------------------------------------------------
            # CHOICE: one label from the supplied categories.
            # ------------------------------------------------
            # Select exactly one category.
            #
            # Useful for routing an agent/workflow.
            # ------------------------------------------------

            "category": Choice(

                instructions=(
                    "Classify the user's query into exactly "
                    "one of the available categories."
                ),

                criteria={
                    "rag": (
                        "The query is primarily about RAG, "
                        "retrieval, embeddings, vector databases, "
                        "or RAG architecture."
                    ),

                    "coding": (
                        "The query primarily asks about writing, "
                        "debugging, or understanding code."
                    ),

                    "system_design": (
                        "The query primarily asks about designing "
                        "software or system architecture."
                    ),

                    "general": (
                        "The query does not clearly belong to "
                        "any of the other categories."
                    ),
                },
            ),
        },
    })


    print("\n" + "=" * 70)
    print("RAW JEV RESPONSE")
    print("=" * 70)

    print(response)


    return {
        "is_simple": response.nouls["is_simple"].noul,
        "category": response.choices["category"].choice,
    }


# ============================================================
# LANGGRAPH
# ============================================================

graph = StateGraph(State)

graph.add_node("analyze_query", analyze_query)

graph.add_edge(START, "analyze_query")
graph.add_edge("analyze_query", END)

app = graph.compile()


if __name__ == "__main__":
    query = "Explain how RAG works with OpenSearch and embeddings"
    result = app.invoke({"query": query})

    print("\n" + "=" * 70)
    print("FINAL LANGGRAPH STATE")
    print("=" * 70)
    print(f"\nQuery:\n{result['query']}")
    print(f"\nIs simple:\n{result['is_simple']}")
    print(f"\nCategory:\n{result['category']}")