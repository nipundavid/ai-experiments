# BM25 vs BM25S Lexical Search Experiments

An experimental playground built to evaluate and visualize the inner workings of the **BM25** information retrieval algorithm. The project utilizes a custom, structurally uniform corpus of 10 authentic Mughlai/Awadhi recipes (*mughlai_recipes*) to stress-test term frequency saturation ($k_1$), document length normalization ($b$), and hierarchical section-level document chunking.

---

## 🚀 Project Overview

Traditional lexical retrieval pipelines often suffer from vertical scrolling friction and document-level noise. This project demonstrates how to transition from **Global Document Retrieval** to **Granular Section Retrieval** by parsing structured Markdown layout assets dynamically. 

By utilizing the highly optimized **`bm25s`** (BM25-Sparse) library, the search system pre-computes sparse lexical score matrices at index time, achieving execution speeds up to 500x faster than legacy native-Python alternatives like `rank-bm25`.

---

## 📂 Repository Structure

```text
bm25s/
│
├── mughlai_recipes/               # The Core Corpus (10 Target Documents)
│   ├── shami_kebab.md
│   ├── mutton_boti_kebab.md
│   ├── mutton_korma.md
│   ├── nihari.md
│   └── ... [remaining recipe markdown files]
│
├── bm25_pipeline.py            # Python Execution Script
├── README.md                   # Project Documentation
└── requirements.txt            # System Dependencies


