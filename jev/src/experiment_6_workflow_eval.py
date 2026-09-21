import os
from typing import TypedDict

from langchain_typesafe import Choice, Noul, TypeSafeClassifier


classifier = TypeSafeClassifier(api_key=os.environ["TYPESAFE_API_KEY"])


class EvaluationCase(TypedDict):
    name: str
    query: str
    expected_category: str
    expected_retrieval: bool


CASES: list[EvaluationCase] = [
    {
        "name": "retrieval question",
        "query": "Which embedding model should I use for semantic search?",
        "expected_category": "rag",
        "expected_retrieval": True,
    },
    {
        "name": "coding question",
        "query": "Why does this Python function raise a KeyError?",
        "expected_category": "coding",
        "expected_retrieval": False,
    },
    {
        "name": "architecture question",
        "query": "How should I design a reliable event-driven system?",
        "expected_category": "system_design",
        "expected_retrieval": False,
    },
]


def evaluate_case(case: EvaluationCase):
    response = classifier.invoke(
        {
            "state": case["query"],
            "questions": {
                "requires_retrieval": Noul(
                    instructions="Does answering this query require external documents or retrieval?"
                ),
                "category": Choice(
                    instructions="Classify the query into exactly one category.",
                    criteria={
                        "rag": "RAG, retrieval, embeddings, or vector databases.",
                        "coding": "Writing, debugging, or understanding code.",
                        "system_design": "Software architecture or system design.",
                        "general": "Anything that does not fit the other categories.",
                    },
                ),
            },
        }
    )

    retrieval = response.nouls["requires_retrieval"].noul >= 0.5
    category = response.choices["category"].choice
    return {
        "category": category,
        "category_confidence": response.choices["category"].confidence,
        "retrieval_probability": response.nouls["requires_retrieval"].noul,
        "category_correct": category == case["expected_category"],
        "retrieval_correct": retrieval == case["expected_retrieval"],
    }


def run_evaluation():
    results = []
    for case in CASES:
        result = evaluate_case(case)
        results.append(result)
        print(
            f"{case['name']}: category={result['category']} "
            f"confidence={result['category_confidence']:.2f} "
            f"retrieval_probability={result['retrieval_probability']:.2f}"
        )

    category_accuracy = sum(item["category_correct"] for item in results) / len(results)
    retrieval_accuracy = sum(item["retrieval_correct"] for item in results) / len(results)
    print(f"Category accuracy: {category_accuracy:.1%}")
    print(f"Retrieval accuracy: {retrieval_accuracy:.1%}")


if __name__ == "__main__":
    run_evaluation()