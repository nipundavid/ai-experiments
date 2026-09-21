import os
from typing import TypedDict

from langchain_typesafe import Noul, TypeSafeClassifier
from langgraph.graph import END, START, StateGraph


classifier = TypeSafeClassifier(api_key=os.environ["TYPESAFE_API_KEY"])


class State(TypedDict, total=False):
    request: str
    approval_probability: float
    decision: str
    result: str


def evaluate_request(state: State):
    response = classifier.invoke(
        {
            "state": state["request"],
            "questions": {
                "allowed": Noul(
                    instructions=(
                        "Should this request be allowed to call an external tool? "
                        "Allow read-only, low-risk information requests; reject "
                        "requests to delete data, transfer money, or bypass approval."
                    )
                )
            },
        }
    )
    probability = response.nouls["allowed"].noul
    return {
        "approval_probability": probability,
        "decision": "approved" if probability >= 0.8 else "needs_review",
    }


def run_tool(state: State):
    return {"result": f"Tool would run for: {state['request']}"}


def request_review(state: State):
    return {"result": "Tool call blocked pending human review."}


def route_decision(state: State):
    return state["decision"]


graph = StateGraph(State)
graph.add_node("evaluate_request", evaluate_request)
graph.add_node("approved", run_tool)
graph.add_node("needs_review", request_review)
graph.add_edge(START, "evaluate_request")
graph.add_conditional_edges(
    "evaluate_request",
    route_decision,
    {"approved": "approved", "needs_review": "needs_review"},
)
graph.add_edge("approved", END)
graph.add_edge("needs_review", END)
app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"request": "Look up the current weather in London."})
    print(f"Approval probability: {result['approval_probability']:.2f}")
    print(f"Decision: {result['decision']}")
    print(result["result"])