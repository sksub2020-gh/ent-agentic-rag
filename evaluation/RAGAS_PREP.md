# RAGAS Evaluation — Preparation Guide

## Overview

RAGAS (Retrieval Augmented Generation Assessment) evaluates your RAG pipeline
automatically — but it needs a **golden dataset** you curate manually first.
This is a one-time setup that pays off every time you change a model, prompt, or retrieval config.

---

## Step 1 — Create the Golden Set

Create `evaluation/golden_set.json` in your project root.

### Format

```json
[
  {
    "question": "What is Docling?",
    "ground_truth": "Docling is an open-source document processing library by IBM that converts PDFs, HTML, and DOCX files into structured representations suitable for LLM pipelines.",
    "reference_contexts": [
      "Optional: paste the exact chunk text the answer should come from."
    ]
  },
  {
    "question": "What version of Docling is documented?",
    "ground_truth": "Version 1.0",
    "reference_contexts": []
  }
]
```

### Fields

| Field | Required | Purpose |
|---|---|---|
| `question` | ✅ | The query sent to the RAG pipeline |
| `ground_truth` | ✅ | The correct answer — written by you |
| `reference_contexts` | ⬜ Optional | Exact chunk(s) the answer should come from — enables `context_recall` scoring |

---

## Step 2 — How Many Questions?

| Corpus size | Minimum | Recommended |
|---|---|---|
| < 50 chunks (your current: 29) | 10 | 15 |
| 50–500 chunks | 20 | 50 |
| 500+ chunks | 50 | 100+ |

RAGAS works with as few as 10 but more questions = more reliable, stable metrics.

---

## Step 3 — Question Variety

Cover all four types to get meaningful scores across all RAGAS metrics:

### Factual
Direct lookup — tests retrieval precision.
```
"What version is documented?"
"Who published this document?"
"What is the minimum token size for chunking?"
```

### Conceptual
Understanding — tests answer relevancy and faithfulness.
```
"What is the purpose of hybrid chunking?"
"How does Docling differ from traditional PDF parsers?"
"What is Reciprocal Rank Fusion?"
```

### Multi-hop
Requires combining information from multiple chunks — tests context recall.
```
"How does Docling handle tables and what format does it output them in?"
"What models are used for embedding and reranking, and why were they chosen?"
```

### Edge Cases
Tests robustness — should return "I don't know" style answers.
```
"What is the pricing model for this product?"  ← not in docs
"Who is the CEO of IBM?"                        ← not in docs
```

---

## Step 4 — What RAGAS Measures

Once the golden set is ready, the evaluator runs each question through your
full pipeline and scores it automatically:

| Metric | What it checks | Needs `reference_contexts`? |
|---|---|---|
| `answer_relevancy` | Is the answer on-topic with the question? | No |
| `faithfulness` | Is every claim in the answer supported by retrieved context? | No |
| `context_precision` | Were the retrieved chunks actually useful for the answer? | No |
| `context_recall` | Did retrieval find the chunks that contain the answer? | ✅ Yes |

> **Tip:** `context_recall` is the most valuable metric for tuning retrieval.
> It requires `reference_contexts` — worth filling in for at least half your questions.

---

## Step 5 — Run Evaluation

Once your golden set is ready:

```bash
python cli/evaluate.py
```

Output will be a scored report saved to `evaluation/results/` showing per-question
and aggregate scores across all four metrics.

---

## Tips for Writing Good Ground Truths

- **Be specific** — "Docling uses HybridChunker" not "it uses a chunker"
- **Match the docs** — copy exact terminology from your source documents
- **Keep it concise** — 1-3 sentences per answer is ideal
- **Don't over-explain** — ground truth is the expected answer, not a full essay
- **For edge cases** — ground truth should be `"This information is not available in the provided documents"`

---

## Example Golden Set (starter for Docling corpus)

```json
[
  {
    "question": "What is Docling?",
    "ground_truth": "Docling is an open-source document conversion library developed by IBM that processes PDFs, HTML, DOCX and other formats into structured representations for LLM pipelines.",
    "reference_contexts": []
  },
  {
    "question": "What chunking strategy does Docling use?",
    "ground_truth": "Docling uses a HybridChunker that combines semantic awareness with token-aware splitting, respecting document structure like headings and tables.",
    "reference_contexts": []
  },
  {
    "question": "What is the capital of France?",
    "ground_truth": "This information is not available in the provided documents.",
    "reference_contexts": []
  }
]
```

---

## File Location

```
rag_project/
└── evaluation/
    ├── golden_set.json        ← you create this
    ├── ragas_evaluator.py     ← already built
    ├── failure_analyzer.py    ← already built
    └── results/               ← auto-created on first run
```
