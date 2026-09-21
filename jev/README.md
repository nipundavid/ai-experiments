# JEv + LangGraph Experiment

A small experimental project exploring **JEv by TypeSafe AI** and how it can be integrated with **LangGraph** to make structured decisions inside an agent workflow.

The goal of this project is not to use JEv as a replacement for an LLM, but to understand how a **decision model** can work alongside generative models and workflow orchestration.

---

## 1. What is JEv?

JEv is a decision-oriented model from **TypeSafe AI**.

Traditional LLM usage typically looks like:

```text
User
  ↓
LLM
  ↓
Generated text
```

JEv is intended for situations where the system needs to make a **bounded, structured decision**:

```text
User
  ↓
JEv
  ↓
Structured decision
```

For example:

```text
"Is this query simple?"
        ↓
       JEv
        ↓
      0.87
```

Instead of asking an LLM to generate a textual explanation and then parsing that explanation, JEv can directly produce a decision that application code can consume.

---

## 2. Why use JEv with LangGraph?

LangGraph is responsible for **workflow orchestration**.

JEv is responsible for **making decisions**.

A generative LLM such as Gemini is responsible for **reasoning and generating content**.

This gives us a useful separation of responsibilities:

```text
                    User
                     │
                     ▼
              ┌─────────────┐
              │     JEv     │
              │  Decision   │
              └──────┬──────┘
                     │
              "What should happen?"
                     │
                     ▼
              ┌─────────────┐
              │  LangGraph  │
              │ Orchestrate │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │    Gemini   │
              │  Reasoning  │
              │ + Generation│
              └─────────────┘
```

A useful mental model is:

> **JEv decides → LangGraph routes → LLM executes**

---

# 3. JEv Decision Primitives

JEv provides different primitives for different types of decisions.

## Noul

`Noul` is useful for a **binary decision**.

Think:

```text
YES / NO
```

Example:

```python
Noul(
    instructions=(
        "Is this a simple question that can be "
        "answered directly without using tools?"
    )
)
```

Conceptually:

```text
Query
  ↓
JEv
  ↓
Noul
  ↓
0.87
```

The application can then interpret the result:

```python
is_simple = response.nouls["is_simple"].noul > 0.5
```

Potential use cases:

- Should we call a tool?
- Does this require human approval?
- Is retrieval necessary?
- Should we escalate?
- Is this request relevant?
- Is the query simple?
- Should an expensive agent be invoked?

---

## Choice

`Choice` is useful when the application needs to select **one category from a predefined set**.

Example:

```python
Choice(
    instructions=(
        "Classify the user's query into exactly "
        "one of the available categories."
    ),
    criteria={
        "rag": "The query is primarily about RAG.",
        "coding": "The query is primarily about coding.",
        "system_design": "The query is about system architecture.",
        "general": "The query does not fit the other categories.",
    },
)
```

Conceptually:

```text
                    Query
                      │
                     JEv
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
         RAG        Coding    System Design
```

This is particularly useful for **agent routing**.

---

## Score

`Score` can be used when a decision is better represented as a numerical evaluation rather than a binary or categorical result.

For example:

```text
How complex is this query?

        ↓
       JEv
        ↓
      0.72
```

Potential use cases:

- Query complexity
- Relevance
- Confidence
- Priority
- Risk
- Quality

The exact constructor/API should be checked against the installed version of `langchain-typesafe`, because the integration is currently evolving.

---

# 4. Current Experiment

The current experiment asks JEv two questions about a user query:

1. Is the query simple?
2. What category does the query belong to?

For example:

```text
"Explain how RAG works with OpenSearch and embeddings"
```

JEv may produce something conceptually like:

```text
is_simple → False

category → rag
```

The exact values are produced by JEv.

---

# 5. Architecture

The current application is intentionally simple:

```text
                 START
                   │
                   ▼
          ┌─────────────────┐
          │   analyze_query │
          │                 │
          │      JEv        │
          └────────┬────────┘
                   │
                   ▼
                  END
```

The LangGraph state contains:

```python
class State(TypedDict):
    query: str
    is_simple: bool
    category: str
```

JEv reads:

```text
query
```

and produces:

```text
is_simple
category
```

LangGraph stores those values in its state.

---

# 6. Installation

This project uses `uv`.

Create the project:

```bash
uv init
```

Install the dependencies:

```bash
uv add langgraph langchain-typesafe
```

If Gemini is also going to be used:

```bash
uv add langchain-google-genai
```

---

# 7. Environment Variables

JEv requires a TypeSafe API key.

Set:

```bash
export TYPESAFE_API_KEY="your-api-key"
```

Or use a `.env` file:

```text
TYPESAFE_API_KEY=your-api-key
```

If using Gemini:

```text
GOOGLE_API_KEY=your-google-api-key
```

The Gemini experiment defaults to `gemini-3.6-flash`. Set `GEMINI_MODEL` if your
Google account exposes a different supported model.

Do not commit `.env` to Git.

Add:

```text
.env
```

to `.gitignore`.

---

# 8. Current Example

A minimal JEv + LangGraph example:

```python
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_typesafe import TypeSafeClassifier, Noul, Choice


# ============================================================
# JEV
# ============================================================

classifier = TypeSafeClassifier(
    api_key=os.environ["TYPESAFE_API_KEY"]
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class State(TypedDict):
    query: str
    is_simple: bool
    category: str


# ============================================================
# JEV NODE
# ============================================================

def analyze_query(state: State):

    query = state["query"]

    response = classifier.invoke({

        # Information JEv should evaluate
        "state": query,

        "questions": {

            # ------------------------------------------------
            # NOUL
            # ------------------------------------------------
            # Binary decision.
            # ------------------------------------------------

            "is_simple": Noul(
                instructions=(
                    "Is this a simple question that can be "
                    "answered directly without retrieval, "
                    "multiple reasoning steps, or external tools?"
                )
            ),

            # ------------------------------------------------
            # CHOICE
            # ------------------------------------------------
            # Categorical decision.
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

    # Inspect JEv during experimentation
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


# ============================================================
# TEST
# ============================================================

query = "Explain how RAG works with OpenSearch and embeddings"

result = app.invoke({
    "query": query
})


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL LANGGRAPH STATE")
print("=" * 70)

print(f"\nQuery:")
print(result["query"])

print(f"\nIs simple:")
print(result["is_simple"])

print(f"\nCategory:")
print(result["category"])
```

---

# 9. The Important Difference From an LLM

A normal LLM call might look like:

```python
response = llm.invoke(
    "Is this query simple?"
)
```

The LLM might return:

```text
Yes, this appears to be a simple query because...
```

The application then has to interpret that response.

With JEv:

```python
response = classifier.invoke(...)
```

The application gets a structured decision.

For example:

```text
Noul
  ↓
0.87
```

or:

```text
Choice
  ↓
"rag"
```

This makes JEv particularly interesting for **control flow**.

---

# 10. JEv + LangGraph Routing

The next natural evolution of this experiment is to use JEv's decision to control LangGraph.

Instead of:

```text
START
  ↓
JEv
  ↓
END
```

we can build:

```text
                         START
                           │
                           ▼
                        ┌─────┐
                        │ JEv │
                        └──┬──┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
             RAG         Coding    System Design
              │            │            │
              ▼            ▼            ▼
          RAG Agent    Code Agent   Design Agent
              │            │            │
              └────────────┼────────────┘
                           ▼
                          END
```

LangGraph supports conditional edges for exactly this type of workflow.

Conceptually:

```python
def route_query(state: State):
    return state["category"]
```

Then:

```python
graph.add_conditional_edges(
    "analyze_query",
    route_query,
    {
        "rag": "rag_agent",
        "coding": "coding_agent",
        "system_design": "system_design_agent",
        "general": "general_agent",
    }
)
```

This creates a clean separation:

```text
JEv
 ↓
Decision

LangGraph
 ↓
Routing

LLM / Agent
 ↓
Execution
```

---

# 11. Why This Is Interesting for Agentic Systems

One common agent architecture is:

```text
LLM
 ↓
Think
 ↓
Decide
 ↓
Tool
 ↓
Think
 ↓
Decide
 ↓
Tool
```

The same LLM is often responsible for both:

1. Open-ended reasoning
2. Simple control decisions

JEv introduces another possibility:

```text
             ┌──────────────┐
             │     JEv      │
             │ Fast Decision│
             └───────┬──────┘
                     │
                     ▼
               LangGraph
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        Agent       RAG       Tool
          │          │          │
          ▼          ▼          ▼
         LLM        LLM        API
```

The idea is to avoid using an expensive generative model for every small decision in an agent loop.

---

# 12. What to Experiment With Next

## Experiment 1 — Binary decisions

Create Noul questions such as:

```text
Does this query require retrieval?

Should we call an external API?

Does this request require human approval?

Is this request relevant to our application?
```

---

## Experiment 2 — Classification

Use Choice:

```text
RAG
Coding
System Design
General
```

Then route the LangGraph workflow based on the result.

---

## Experiment 3 — Multiple decisions in one call

Ask JEv several questions simultaneously:

```text
Is this simple?
Does this require retrieval?
Is this coding related?
Which category is it?
```

Observe how structured the response is.

---

## Experiment 4 — JEv + Gemini

Build:

```text
User
 │
 ▼
JEv
 │
 ├── Simple ────────► Gemini
 │
 ├── RAG ───────────► RAG Agent
 │
 ├── Coding ────────► Coding Agent
 │
 └── System Design ─► Design Agent
```

This is where the experiment becomes a realistic agent architecture.

---

## Experiment 5 — JEv as a guard/decision layer

Try:

```text
User
 ↓
JEv
 ↓
Should this action be allowed?
 ↓
YES ──► Tool
NO  ──► Stop
```

This is a useful pattern for:

- Tool authorization
- Human escalation
- Expensive-model routing
- Retrieval decisions
- Workflow branching
- Agent termination

---

# 13. Key Takeaways

The main thing to remember from this experiment:

```text
LLM
=
Open-ended generation and reasoning
```

```text
JEv
=
Fast structured decisions
```

```text
LangGraph
=
State + workflow + orchestration
```

Together:

```text
        ┌──────────────┐
        │     JEv      │
        │    Decide    │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  LangGraph   │
        │    Route     │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ LLM / Agent  │
        │    Execute   │
        └──────────────┘
```

The purpose of this repository is to experiment with this separation and understand where a specialized decision model can fit into modern agent architectures.

---

## Useful Commands

Run the application:

```bash
python src/main.py
```

Run the experiments:

```bash
python src/experiment_1.py  # Noul + Choice in one call
python src/experiment_2_score.py  # Ordered complexity score
python src/experiment_3_routing.py  # Conditional category routing
python src/experiment_4_gemini.py  # JEv classification + Gemini
python src/experiment_5_guard.py  # Tool approval guard
```

Or:

```bash
uv run python src/main.py
```

Inspect the installed JEv question schemas:

```bash
python -c "from langchain_typesafe import Choice, Score, Noul; print('CHOICE:', Choice.model_fields); print('SCORE:', Score.model_fields); print('NOUL:', Noul.model_fields)"
```

Check installed packages:

```bash
uv pip list
```

Update dependencies:

```bash
uv lock
```

---

## Status

Current:

- [x] JEv installed
- [x] JEv API key configured
- [x] LangGraph integration
- [x] Noul experiment
- [x] Choice experiment
- [x] Structured JEv output
- [x] Score experiment
- [x] Conditional LangGraph routing
- [x] JEv + Gemini
- [x] Multi-agent routing (worker stubs in `experiment_3_routing.py`)
- [x] Tool-selection experiment (approval gate in `experiment_5_guard.py`)
- [x] JEv-based guard/approval workflow
