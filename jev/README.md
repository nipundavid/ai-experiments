# Jev + LangGraph Experiment

A small experimental project exploring **Jev by TypeSafe AI** and how it can be integrated with **LangGraph** to make structured decisions inside an agent workflow.

The goal of this project is not to use Jev as a replacement for an LLM, but to understand how a **decision model** can work alongside generative models and workflow orchestration.

---

## Official Resources and Related Articles

### TypeSafe AI and Jev

- [TypeSafe AI](https://typesafe.ai/): official TypeSafe AI website.
- [Introducing System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev): TypeSafe AI's overview of Jev and structured decisions.

### LangGraph and LangChain

- [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview): official concepts, guides, and API documentation.
- [LangGraph GitHub repository](https://github.com/langchain-ai/langgraph): source code and examples.
- [Building a Harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev): LangChain's example of using Jev in an agent harness.

### Google Gemini

- [Gemini API documentation](https://ai.google.dev/gemini-api/docs): official Gemini API guides and reference.
- [Gemini models](https://ai.google.dev/gemini-api/docs/models): available models and model capabilities.
- [Google AI for Developers blog](https://developers.googleblog.com/): announcements and technical posts from Google's AI developer team.
- [ChatGoogleGenerativeAI integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai): LangChain documentation used by `experiment_4_gemini.py`.

### Other Jev Use Cases

- [12 Jev Use Cases Tested](https://www.mindstudio.ai/blog/jev-use-cases-automation): automation and high-volume classification examples.
- [What is Jev?](https://vercel.com/i/what-is-jev): bounded decisions, evidence, confidence, and human review.

---

## Jev in Brief

Jev is a decision-focused model from TypeSafe AI. It is designed to take
application state plus explicitly defined questions and return structured
decisions that software can use directly.

Jev is not intended to replace a generative LLM. The central idea of this
project is to use the right model for each part of a workflow:

```text
Application state
    ↓
   Jev            → decides: classify, score, approve, or defer
    ↓
   LangGraph      → stores state and routes the workflow
    ↓
  LLM / tool      → explains, generates, or performs an action
```

### What Goes In and What Comes Out

| Part        | Meaning                                                              | Example in this project                                                       |
| ----------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `state`     | The text or structured data Jev evaluates                            | A user query or tool request                                                  |
| `questions` | The bounded decisions the application wants answered                 | `is_simple`, `category`, or `requires_retrieval`                              |
| `Noul`      | A yes/no-style probability                                           | Approve a tool call in [`experiment_5_guard.py`](src/experiment_5_guard.py)   |
| `Choice`    | One label from predefined options, with probabilities and confidence | Route to a worker in [`experiment_3_routing.py`](src/experiment_3_routing.py) |
| `Score`     | A value on an ordered rubric                                         | Estimate complexity in [`experiment_2_score.py`](src/experiment_2_score.py)   |

The application defines the decision space first. Jev evaluates the supplied
state against that space and returns typed answers, rather than a paragraph
that the application must parse.

### Jev Compared With a Generative LLM

| Concern                 | Generative LLM                                                                       | Jev                                                            |
| ----------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Primary job             | Explain, reason in depth, write, or generate content                                 | Make bounded decisions for software                            |
| Output                  | Usually free-form text or generated tool arguments                                   | Typed `Noul`, `Choice`, or `Score` answers                     |
| Best fit                | Answers, summaries, code, plans, and creative work                                   | Classification, routing, scoring, filtering, and approval      |
| Application control     | The application interprets generated text or tool calls                              | The application defines options, thresholds, and branches      |
| Uncertainty             | May require additional handling or log-probability tooling                           | Probabilities and confidence are part of the decision response |
| Role in this repository | Generates the final answer in [`experiment_4_gemini.py`](src/experiment_4_gemini.py) | Makes the routing and policy decisions in experiments 1–6      |

This is a division of responsibilities, not a claim that Jev is universally
more accurate or a replacement for an LLM. The evaluation experiment exists to
measure whether a particular decision workflow is reliable enough for its use
case.

### Why This Matters

Using a generative model for every small control-flow decision can make a
workflow harder to validate: the application has to interpret prose, enforce
allowed values, and decide what to do with uncertainty. Jev makes those
decisions explicit so ordinary application code can apply policies such as:

```python
if approval_probability >= 0.8:
  run_tool()
else:
  request_human_review()
```

The experiments demonstrate the progression:

1. [`experiment_1.py`](src/experiment_1.py) shows multiple independent decisions in one request.
2. [`experiment_2_score.py`](src/experiment_2_score.py) shows when an ordered score is more useful than a label.
3. [`experiment_3_routing.py`](src/experiment_3_routing.py) connects a decision to LangGraph branches.
4. [`experiment_4_gemini.py`](src/experiment_4_gemini.py) hands generation to Gemini after classification.
5. [`experiment_5_guard.py`](src/experiment_5_guard.py) turns probability into an approval policy.
6. [`experiment_6_workflow_eval.py`](src/experiment_6_workflow_eval.py) checks the workflow across representative cases.

---

## Goals and Takeaways

This project is trying to answer a practical question:

> Can a specialized decision model make agent workflows simpler, more
> predictable, and easier to control than asking a generative LLM to make every
> decision?

The experiments are intended to show that:

- **Jev is useful for bounded decisions:** use `Noul`, `Choice`, and `Score` when the output can be expressed as a probability, category, or numeric scale.
- **Structured decisions simplify application code:** the workflow can consume typed results directly instead of parsing free-form LLM text.
- **LangGraph should own orchestration:** Jev decides, while LangGraph stores state and routes execution.
- **Probabilities can become explicit policy:** confidence and decision probabilities can control thresholds, fallbacks, and human review.
- **Decision questions should be decomposed:** several narrow questions are easier to inspect and evaluate than one broad prompt asking an LLM to do everything.
- **Insufficient evidence is a valid result:** a workflow should be able to defer or escalate instead of forcing an unreliable category.
- **Generative models still have a separate role:** Gemini is used for explanation and generation after Jev has made a control-flow decision.
- **The workflow must be evaluated, not assumed correct:** speed and type safety do not prove semantic accuracy, so repeated evaluation cases are included.

### Experiment Map

| Experiment                      | What we are trying to learn                                  | Main takeaway                                                                             |
| ------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `experiment_1.py`               | Can one Jev call answer multiple structured questions?       | Multiple independent decisions can be returned in one typed response.                     |
| `experiment_2_score.py`         | When is a continuous score more useful than a yes/no answer? | Ordered rubrics represent complexity and other gradual properties better than categories. |
| `experiment_3_routing.py`       | Can Jev control LangGraph branches?                          | A `Choice` can select a workflow worker, including an insufficient-evidence path.         |
| `experiment_4_gemini.py`        | Where should Jev end and a generative model begin?           | Jev handles routing; Gemini handles the final natural-language answer.                    |
| `experiment_5_guard.py`         | Can probabilities enforce a tool-approval policy?            | Thresholds can approve low-risk work and send uncertain requests for review.              |
| `experiment_6_workflow_eval.py` | How do we test a decision workflow across cases?             | A working demo still needs repeatable accuracy and confidence measurements.               |

---

## 1. What is Jev?

Jev is a decision-oriented model from **TypeSafe AI**.

Traditional LLM usage typically looks like:

```text
User
  ↓
LLM
  ↓
Generated text
```

Jev is intended for situations where the system needs to make a **bounded, structured decision**:

```text
User
  ↓
Jev
  ↓
Structured decision
```

For example:

```text
"Is this query simple?"
        ↓
       Jev
        ↓
      0.87
```

Instead of asking an LLM to generate a textual explanation and then parsing that explanation, Jev can directly produce a decision that application code can consume.

---

## 2. Why use Jev with LangGraph?

LangGraph is responsible for **workflow orchestration**.

Jev is responsible for **making decisions**.

A generative LLM such as Gemini is responsible for **reasoning and generating content**.

This gives us a useful separation of responsibilities:

```text
                    User
                     │
                     ▼
              ┌─────────────┐
              │     Jev     │
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

> **Jev decides → LangGraph routes → LLM executes**

---

# 3. Jev Decision Primitives

Jev provides different primitives for different types of decisions.

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
Jev
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
                     Jev
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
       Jev
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

The current experiment asks Jev two questions about a user query:

1. Is the query simple?
2. What category does the query belong to?

For example:

```text
"Explain how RAG works with OpenSearch and embeddings"
```

Jev may produce something conceptually like:

```text
is_simple → False

category → rag
```

The exact values are produced by Jev.

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
          │      Jev        │
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

Jev reads:

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

Jev requires a TypeSafe API key.

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

A minimal Jev + LangGraph example:

```python
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_typesafe import TypeSafeClassifier, Noul, Choice


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
    is_simple: bool
    category: str


# ============================================================
# Jev NODE
# ============================================================

def analyze_query(state: State):

    query = state["query"]

    response = classifier.invoke({

        # Information Jev should evaluate
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

    # Inspect Jev during experimentation
    print("\n" + "=" * 70)
    print("RAW Jev RESPONSE")
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

With Jev:

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

This makes Jev particularly interesting for **control flow**.

---

# 10. Jev + LangGraph Routing

The next natural evolution of this experiment is to use Jev's decision to control LangGraph.

Instead of:

```text
START
  ↓
Jev
  ↓
END
```

we can build:

```text
                         START
                           │
                           ▼
                        ┌─────┐
                        │ Jev │
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
Jev
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

Jev introduces another possibility:

```text
             ┌──────────────┐
             │     Jev      │
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

Ask Jev several questions simultaneously:

```text
Is this simple?
Does this require retrieval?
Is this coding related?
Which category is it?
```

Observe how structured the response is.

---

## Experiment 4 — Jev + Gemini

Build:

```text
User
 │
 ▼
Jev
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

## Experiment 5 — Jev as a guard/decision layer

Try:

```text
User
 ↓
Jev
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

---

## Experiment 6 — Workflow evaluation

Run the same decision workflow over several representative cases and record:

- Category accuracy
- Retrieval-decision accuracy
- Choice confidence
- Noul retrieval probability

This follows the harness and workflow-evaluation pattern described in the
[LangChain Jev article](https://www.langchain.com/blog/building-a-harness-with-jev).

# 13. Key Takeaways

The main thing to remember from this experiment:

```text
LLM
=
Open-ended generation and reasoning
```

```text
Jev
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
        │     Jev      │
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
python src/experiment_4_gemini.py  # Jev classification + Gemini
python src/experiment_5_guard.py  # Tool approval guard
python src/experiment_6_workflow_eval.py  # Repeatable workflow evaluation
```

Or:

```bash
uv run python src/main.py
```

Inspect the installed Jev question schemas:

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

- [x] Jev installed
- [x] Jev API key configured
- [x] LangGraph integration
- [x] Noul experiment
- [x] Choice experiment
- [x] Structured Jev output
- [x] Score experiment
- [x] Conditional LangGraph routing
- [x] Jev + Gemini
- [x] Multi-agent routing (worker stubs in `experiment_3_routing.py`)
- [x] Tool-selection experiment (approval gate in `experiment_5_guard.py`)
- [x] Jev-based guard/approval workflow
- [x] Workflow evaluation harness
