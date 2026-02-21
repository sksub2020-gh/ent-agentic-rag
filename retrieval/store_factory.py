"""
Store factory — single source of truth for backend selection.

All entrypoints (ingestion, rag_query, app.py, graph.py) import from here.
Adding a new backend = one new _build_x() function + one registry entry.

Controlled by STORE_BACKEND in .env:
    supabase  → SupabaseStore  (production)
    qdrant    → QdrantStore    (local dev, native hybrid)
    milvus    → MilvusLiteStore + BM25SStore (fallback)
"""

import logging

logger = logging.getLogger(__name__)


# ── Backend builders ──────────────────────────────────────────────────────────


def _build_supabase(embedder):
    from retrieval.supabase_store import SupabaseStore

    store = SupabaseStore()
    return store, store


def _build_qdrant(embedder):
    from retrieval.qdrant_store import QdrantStore

    store = QdrantStore(dimension=embedder.dimension)
    return store, store


def _build_milvus(embedder):
    from retrieval.milvus_store import MilvusLiteStore
    from retrieval.bm25_store import BM25SStore

    return MilvusLiteStore(dimension=embedder.dimension), BM25SStore()


BACKENDS = {
    "supabase": _build_supabase,
    "qdrant": _build_qdrant,
    "milvus": _build_milvus,
}


# ── Public API ────────────────────────────────────────────────────────────────


def build_stores(embedder):
    """
    Returns (vector_store, sparse_store) for the configured backend.
    Used by ingestion pipeline — stores only, no retriever.
    """
    from config.settings import config

    builder = BACKENDS.get(config.store_backend)
    if not builder:
        raise ValueError(
            f"Unknown STORE_BACKEND: '{config.store_backend}'. "
            f"Choose from: {list(BACKENDS)}"
        )
    logger.info(f"Store backend: {config.store_backend}")
    return builder(embedder)


def build_retriever(embedder=None):
    """
    Returns a fully configured HybridRetriever for the configured backend.
    Used by rag_query, app.py, graph.py — anything that queries.
    Pass an existing embedder to avoid loading the model twice.
    """
    from ingestion.embedder import MpetEmbedder
    from retrieval.hybrid_retriever import HybridRetriever, FlashRankReranker

    embedder = embedder or MpetEmbedder()
    vector_store, sparse_store = build_stores(embedder)

    return HybridRetriever(
        vector_store=vector_store,
        sparse_store=sparse_store,
        embedder=embedder,
        reranker=FlashRankReranker(),
    )


def build_pipeline():
    """
    Returns (llm, retriever) — full pipeline ready for querying.
    Used by app.py and agentic_rag.py.
    """
    from core.llm_client import LLMClient
    from ingestion.embedder import MpetEmbedder

    embedder = MpetEmbedder()
    retriever = build_retriever(embedder=embedder)
    llm = LLMClient()
    return llm, retriever
