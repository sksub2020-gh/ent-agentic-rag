# Enterprise RAG System — Project Summary

## What We Built

A production-grade **Retrieval Augmented Generation (RAG)** system with an agentic
pipeline, multi-backend support, and a Streamlit UI. Built incrementally over multiple
sessions — every decision was deliberate and understood.

---

## Architecture

```
Documents (PDF/HTML/DOCX)
    ↓
Docling HybridChunker        — structure-aware chunking (respects headings, tables)
    ↓
MpetEmbedder                 — dense vectors (all-mpnet-base-v2, 768d)
    ↓
Store (config-driven)        — vectors + sparse index
    ↓
HybridRetriever              — dense + sparse + RRF fusion
    ↓
FlashRank                    — cross-encoder reranking
    ↓
Agentic Graph (LangGraph)
    ├── Router Node           — rag vs direct answer
    ├── RAG Node              — retrieval + generation
    ├── Critique Node         — grounding check + retry
    └── Guard Nodes           — input/output safety
    ↓
Mistral-7B via Ollama        — local LLM, fully offline
    ↓
Streamlit UI                 — chat interface, linear/agentic toggle
```

---

## Multi-Backend Store (plug-n-play)

One of the strongest design decisions — abstract interfaces mean the entire retrieval
stack is swappable via a single `.env` change:

| Backend | Mode | Dense | Sparse | Fusion |
|---|---|---|---|---|
| **Supabase** | Production | pgvector | tsvector | SQL RRF (native) |
| **Qdrant** | Local dev | Vectors | BM42/FastEmbed | Qdrant RRF (native) |
| **Milvus + BM25S** | Fallback | Milvus-Lite | BM25S | Python RRF |

```bash
# Switch backends — zero code changes
STORE_BACKEND=supabase   # production
STORE_BACKEND=qdrant     # local dev
STORE_BACKEND=milvus     # offline fallback
```

---

## Project Structure

```
rag_project/
├── cli/                          # Entrypoints (no __init__.py)
│   ├── app.py                    # streamlit run cli/app.py
│   ├── rag_query.py              # python cli/rag_query.py "question"
│   ├── ingestion_pipeline.py     # python cli/ingestion_pipeline.py ./docs/
│   ├── agentic_rag.py            # python cli/agentic_rag.py (REPL)
│   └── evaluate.py               # python cli/evaluate.py
├── core/
│   ├── interfaces.py             # Abstract base classes (VectorStoreBase etc.)
│   └── llm_client.py             # OpenAI-compatible LLM client
├── ingestion/
│   ├── docling_chunker.py        # Docling HybridChunker wrapper
│   └── embedder.py               # MpetEmbedder (dense vectors)
├── retrieval/
│   ├── store_factory.py          # Registry — single source of truth for backends
│   ├── supabase_store.py         # Supabase pgvector + tsvector
│   ├── qdrant_store.py           # Qdrant dense + BM42 sparse
│   ├── milvus_store.py           # Milvus-Lite dense
│   ├── bm25_store.py             # BM25S sparse
│   ├── sqlite_sparse_store.py    # SQLite FTS5 sparse (alternative to BM25S)
│   └── hybrid_retriever.py       # RRF fusion + FlashRank reranking
├── agents/
│   ├── state.py                  # AgentState TypedDict
│   ├── graph.py                  # LangGraph graph builder
│   ├── router_node.py            # Route: rag vs direct
│   ├── rag_node.py               # Retrieve + generate
│   └── critique_node.py          # Grounding check + retry
├── guardrails/
│   ├── guard_runner.py           # GuardRunner orchestrator
│   ├── input_guards.py           # Injection, PII, topic, length
│   └── output_guards.py          # Toxicity, hallucination, PII redaction
├── evaluation/
│   ├── ragas_evaluator.py        # RAGAS evaluation runner
│   ├── failure_analyzer.py       # Per-question failure analysis
│   └── golden_set.json           # ← you create this
├── config/
│   └── settings.py               # Pydantic-settings, all config
├── pyproject.toml                # pip install -e . makes root importable
└── .env                          # All secrets and config
```

---

## Key Design Patterns

### Abstract Interfaces (`core/interfaces.py`)
`VectorStoreBase`, `SparseStoreBase`, `EmbedderBase`, `RerankerBase` — every component
is swappable. Adding Pinecone = one new file implementing the interface.

### Store Factory (`retrieval/store_factory.py`)
Registry pattern — `build_pipeline()`, `build_retriever()`, `build_stores()`.
All entrypoints call one function. Adding a new backend = one new `_build_x()` function
plus one registry entry. Nothing else changes.

```python
BACKENDS = {
    "supabase": _build_supabase,
    "qdrant":   _build_qdrant,
    "milvus":   _build_milvus,
    # "pinecone": _build_pinecone,  ← future, one line
}
```

### Auto-detected Retrieval Path (`HybridRetriever`)
```
hasattr(store, "hybrid_search") → Path A: SQL/native RRF (Supabase, Qdrant)
else                            → Path B: Python RRF (Milvus + BM25S)
```
FlashRank reranks on top of whichever path produced the candidates.

### Config-driven Everything (`config/settings.py`)
Pydantic-settings — all backends, models, and parameters controlled via `.env`.
Each nested config declares `env_file` + `extra="ignore"` explicitly.

### CLI Entrypoints (`cli/`)
No `__init__.py` — scripts folder, not a package. `pyproject.toml` with
`pip install -e .` / `uv pip install -e .` makes the project root importable
from any entrypoint regardless of working directory.

---

## Hybrid Search — How It Works

Three stages, regardless of backend:

**1. Dense search** — cosine similarity on embedding vectors.
Good at paraphrasing, synonyms, semantic meaning.
Bad at exact keywords, version numbers, codes.

**2. Sparse search** — term frequency matching (BM25 / tsvector / BM42).
Good at exact matches, keywords, IDs.
Bad at paraphrasing and synonyms.

**3. RRF Fusion** — Reciprocal Rank Fusion combines both lists.
`score = Σ 1/(k+rank)` — chunks appearing in both lists get a bonus.
`k=60` prevents top ranks from dominating.

**4. FlashRank** — cross-encoder sees query + chunk together.
Reranks fused candidates down to `top_k_rerank` (default 5).
Local, no API cost, fully offline.

---

## Agentic Pipeline (LangGraph)

```
input_guard → router → rag → critique → output_guard → END
                        ↑______↓ (retry if not grounded, max 2x)
```

| Node | Reads | Writes | Decision |
|---|---|---|---|
| `input_guard` | query | blocked, warnings | Hard block or pass |
| `router` | query | route, reasoning | `rag` or `direct` |
| `rag` | query, route, retry_count | chunks, context, answer | Retrieves + generates |
| `critique` | query, context, answer | grounded, reasoning | Pass or retry |
| `output_guard` | answer | warnings, redactions | Final safety check |

**Retry logic** — on failed grounding, query is expanded with
`"(provide more detail and related context)"` before re-retrieval.
Max 2 retries, then disclaimer appended and pipeline exits.

---

## Streamlit UI

```
streamlit run cli/app.py
```

**Sidebar:**
- Ollama connectivity status (cached 30s)
- Pipeline mode toggle: `⚡ Linear RAG` vs `🤖 Agentic RAG`
- Show sources / chunk content / agent trace toggles
- Clear chat button
- Ingestion command hint

**Agentic mode extras per response:**
- Route chosen + router reasoning
- Grounded ✅ / ❌ + critique reasoning
- Retry count
- Guard warnings / blocks

**Linear mode** — fast path, no guardrails, no critique. Same retrieval stack.

Both modes share the same cached `llm`, `retriever`, and `graph` instances —
switching modes doesn't reload anything.

---

## Good Points

**Unified store** — Supabase and Qdrant each replace two separate stores
(Milvus + BM25S). One connection, one client, native hybrid search.

**SQL RRF in Supabase** — one round trip instead of two searches + Python fusion.
Postgres handles the merge natively via CTE.

**Qdrant BM42** — neural sparse vectors via FastEmbed. Quality upgrade over
BM25 term frequency. Lazy-loaded and cached on the store instance — no repeated
model loading per query.

**FlashRank** — local cross-encoder, no API cost, no internet. Sits on top of
any retrieval path — the interface doesn't care what produced the candidates.

**`@st.cache_resource`** — pipeline built once per session. Both linear and
agentic share the same instances — no duplicate loading when toggling modes.

**Open/closed principle** — adding a new backend, embedder, or reranker never
requires modifying existing code — only adding new files and registry entries.

---

## Lessons Learned

**Pydantic nested configs don't inherit `env_file`**
Each nested `BaseSettings` must declare `env_file` + `extra="ignore"` explicitly.
Without it, nested configs only read process environment — not the `.env` file —
causing silent fallback to defaults (e.g. empty connection string → local socket).

**Python has no method overloading**
Defining `search()` twice silently overwrites the first definition. Caused a
`"function plainto_tsquery(unknown, numeric[])"` Postgres error — the wrong
`search()` was being called with a float array. Solution: explicit naming
(`search_dense`, `search_sparse`).

**psycopg2 auto-deserializes JSONB**
`json.loads()` on an already-parsed dict throws `"must be str, bytes or bytearray, not dict"`.
Always check `isinstance(value, str)` before calling `json.loads()`.

**BM25S loads corpus into memory on every instantiation**
Fine for small corpora, a startup cost problem at scale. Supabase `tsvector` and
SQLite FTS5 solve this — the index lives on disk, queries hit it directly.

**LlamaIndex is glue, not capability**
`QueryFusionRetriever` fuses retrievers you explicitly provide — it doesn't conjure
sparse search on stores that don't support it natively. Qdrant and Supabase do hybrid
natively; Milvus-Lite and Chroma don't, regardless of framework wrapper.

**`encode_sparse()` is version-dependent**
`QdrantClient.encode_sparse()` only exists in newer client versions. Calling
FastEmbed's `SparseTextEmbedding` directly is more portable and version-stable.
Cache the encoder on the store instance to avoid reloading per query.

**Table noise is expected, not a bug**
Docling correctly extracts table rows as chunks — data rows without headers become
orphaned number sequences. They score low on FlashRank and rarely pollute answers.
Accept as a known limitation; revisit if RAGAS `context_precision` flags it.

**Streamlit ternary leaks DeltaGenerator**
`st.success(...) if ok else st.error(...)` evaluates both branches and the return
value renders as raw object internals in the UI. Always use `if/else` blocks for
Streamlit UI calls.

**Postgres pooler URL more reliable than direct connection**
Direct Supabase host (`db.[ref].supabase.co`) may not resolve on all networks.
Session pooler URL (`aws-0-[region].pooler.supabase.com:5432`) is more reliable.

---

## Retrieval Store Comparison

| | Supabase | Qdrant | Milvus-Lite | Elasticsearch |
|---|---|---|---|---|
| Dense | ✅ pgvector | ✅ Native | ✅ Native | ✅ Native |
| Sparse | ✅ tsvector | ✅ BM42 | ❌ Need BM25S | ✅ Native |
| Hybrid | ✅ SQL RRF | ✅ Native RRF | ❌ Python RRF | ✅ Native |
| Scalar filter | ✅ SQL WHERE | ✅ Payload | ⚠️ Limited | ✅ Native |
| Local file | ❌ Cloud only | ✅ | ✅ | ❌ |
| Self-hosted | ✅ | ✅ | ✅ | ✅ |
| Ops complexity | Low | Low | Very low | High |

---

## What's Next

- **RAGAS evaluation** — build `evaluation/golden_set.json` (10-15 questions),
  run `cli/evaluate.py`, measure `faithfulness`, `context_recall`,
  `context_precision`, `answer_relevancy`
- **Better embedder** — `BAAI/bge-large-en-v1.5` (free, better MTEB) or
  `text-embedding-3-small` (OpenAI, paid, best quality) — both require re-ingestion
- **Pinecone backend** — `_build_pinecone()` in store factory, one `PineconeStore` file
- **Query expansion** — already stubbed in RAG node retry path, promote to first-class
- **Streaming** — `llm.stream()` + Streamlit `st.write_stream()` for token-by-token output
