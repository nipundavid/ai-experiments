# AI Experiments

A collection of small, focused experiments exploring information retrieval,
structured AI decisions, and agent workflow orchestration.

## Modules

| Module                   | Focus                                  | What has been done                                                                                                                                                                                                                 |
| ------------------------ | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [bm25s](bm25s/README.md) | Lexical information retrieval          | Built a BM25S search pipeline over a Mughlai recipe corpus. The experiment loads Markdown documents, tokenizes and stems them, builds a Lucene-style BM25 index, and compares ranked results for several queries.                  |
| [jev](jev/README.md)     | Structured decisions and agent routing | Integrated TypeSafe AI JEv with LangGraph. The experiments cover binary decisions with `Noul`, category selection with `Choice`, ordered evaluation with `Score`, conditional routing, Gemini execution, and tool-approval guards. |

## Module Index

### [BM25S](bm25s/README.md)

Explores fast lexical search using the `bm25s` library and a corpus of
Mughlai/Awadhi recipe documents.

- [Pipeline](bm25s/bm25_pipeline.py): loads the recipe corpus and runs ranked search queries.
- [Recipe corpus](bm25s/mughlai_recipes/): Markdown documents used as the search collection.
- [Project README](bm25s/README.md): notes on BM25 scoring, chunking, and the experiment design.

### [JEv + LangGraph](jev/README.md)

Explores how a decision-oriented model can make bounded, structured decisions
while LangGraph manages state and workflow execution.

- [Experiment 1](jev/src/experiment_1.py): evaluates a query with `Noul` and `Choice` in one JEv call.
- [Experiment 2](jev/src/experiment_2_score.py): scores query complexity against an ordered rubric.
- [Experiment 3](jev/src/experiment_3_routing.py): routes queries to category-specific LangGraph workers.
- [Experiment 4](jev/src/experiment_4_gemini.py): combines JEv classification with Gemini response generation.
- [Experiment 5](jev/src/experiment_5_guard.py): gates a simulated tool call using a JEv approval decision.
- [Project README](jev/README.md): detailed concepts, setup instructions, and experiment notes.

## Repository Status

Both modules are working experimental projects. They are intended as learning
and prototyping environments rather than production-ready systems.
