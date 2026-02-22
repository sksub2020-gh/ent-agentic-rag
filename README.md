# Enterprise RAG Project

## Stack
- LLM: Mistral-7B via Ollama (OpenAI-compatible)
- VectorDB: Milvus-Lite (swap-ready via interface)
- Chunking: IBM Docling (Hybrid Chunker)
- Embeddings: mpeT (sentence-transformers)
- Sparse Search: BM25S
- Reranker: FlashRank
- Agents: LangGraph
- Tracing: LangSmith
- Evaluation: RAGAS

## Project Structure
```
rag_project/
├── config/
│   └── settings.py          # All config in one place
├── core/
│   ├── interfaces.py        # Abstract base classes (plug-n-play contracts)
│   └── llm_client.py        # OpenAI-compatible Ollama client
├── ingestion/
│   ├── docling_chunker.py   # IBM Docling hybrid chunker
│   └── embedder.py          # mpeT dense embeddings
├── retrieval/
│   ├── milvus_store.py      # Milvus-Lite vector store (swappable)
│   ├── bm25_store.py        # BM25S sparse index (swappable)
│   └── hybrid_retriever.py  # RRF fusion + FlashRank reranking
├── agents/                  # Phase 4 - LangGraph agents
└── evaluation/              # Phase 5 - RAGAS evaluation
```

## Design Principles
- **Modular**: Each component is independent and testable
- **Plug-n-play**: Swap Milvus → Pinecone by implementing `VectorStoreBase`
- **OpenAI-compatible**: All LLM calls use standard OpenAI SDK against Ollama


## Run app
```
pip install -r requirements.txt
cp .env.example .env          # fill in LangSmith key
ollama serve                  # in a separate terminal
ollama pull mistral

# Ingest a doc
python ingestion_pipeline.py ./your_doc.pdf

# Query it
python rag_query.py "What does this document say about X?"
```

## Phse 4 - Run Agents
```
# Single query
python agentic_rag.py "What does the policy say about returns?"

# Interactive REPL
python agentic_rag.py

# Tests (no services needed)
python -m pytest agents/tests/ -v
```

### Eval with RAGAs
here's the mental model:
- golden_set.py — bootstraps your test set. Auto-generates Q&A pairs from your ingested chunks using the LLM, exports a CSV so you can review/edit before trusting it. Always human-review the CSV — auto-generated ground truth has errors.
- ragas_evaluator.py — runs all 4 metrics (faithfulness, answer_relevancy, context_precision, context_recall) against the full agentic pipeline. Uses Ollama as the evaluation LLM — fully local, no OpenAI needed.
- failure_analyzer.py — the practical part. Maps each failing metric to exact root causes and specific lines/configs to change. Includes a per-sample pattern detector (retry rate, router over-triggering, question type breakdown).
- evaluate.py — CLI that ties it all together.
- One thing to note: RAGAS's context_recall metric needs ground truth contexts (what chunks should have been retrieved), which auto-generation approximates. For truly reliable recall scores, hand-curating at least 10-15 samples in the CSV pays off.

### The Workflow
```
# Step 1 — ingest your docs first
python ingestion_pipeline.py ./your_docs/

# Step 2 — build golden set, review the CSV, edit bad Q&A pairs
python evaluate.py build
# → edit evaluation/golden_set.csv, then re-save as golden_set.json

# Step 3 — run eval
python evaluate.py run

# Step 4 — get fix recommendations
python evaluate.py analyze

# Or all in one
python evaluate.py all
```

'''
pip show ragas langchain langchain-openai datasets 2>/dev/null | grep -E "Name|Version"
'''