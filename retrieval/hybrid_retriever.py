"""
Hybrid Retriever — fuses dense + sparse results via RRF, then reranks with FlashRank.
This is the core retrieval pipeline for Phase 1 & 2.
"""
import logging
from flashrank import Ranker, RerankRequest

from core.interfaces import (
    VectorStoreBase, SparseStoreBase, EmbedderBase, RerankerBase, RetrievedChunk
)
from config.settings import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FlashRank Reranker (implements RerankerBase)
# ---------------------------------------------------------------------------

class FlashRankReranker(RerankerBase):
    """Local FlashRank reranker — no API calls, runs fully offline."""

    def __init__(self):
        self.ranker = Ranker(model_name=config.retrieval.reranker_model)
        logger.info(f"FlashRankReranker ready — model: {config.retrieval.reranker_model}")

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []

        passages = [
            {"id": i, "text": rc.chunk.content}
            for i, rc in enumerate(chunks)
        ]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        reranked = []
        for r in results[:top_k]:
            original = chunks[r["id"]]
            reranked.append(RetrievedChunk(
                chunk=original.chunk,
                score=r["score"],
                source="reranked",
            ))

        logger.debug(f"Reranked {len(chunks)} → {len(reranked)} chunks")
        return reranked


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion — combines dense + sparse rankings.
    Score = Σ 1 / (k + rank)
    k=60 is the standard constant (Robertson et al.)
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, rc in enumerate(dense_results, start=1):
        cid = rc.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = rc

    for rank, rc in enumerate(sparse_results, start=1):
        cid = rc.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = rc

    # Sort by fused score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    fused = []
    for cid in sorted_ids:
        rc = chunk_map[cid]
        fused.append(RetrievedChunk(
            chunk=rc.chunk,
            score=scores[cid],
            source="rrf_fused",
        ))

    return fused


# ---------------------------------------------------------------------------
# Hybrid Retriever (orchestrates everything)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Full retrieval pipeline with two paths:

    Path A — Supabase (SQL-native RRF):
      1. hybrid_search() — dense + sparse + RRF in one SQL query
      2. FlashRank reranking

    Path B — Milvus + BM25S (Python-side RRF):
      1. Dense search (Milvus)
      2. Sparse search (BM25S)
      3. Python RRF fusion
      4. FlashRank reranking

    Path is auto-detected: if vector_store has hybrid_search(), use Path A.
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        sparse_store: SparseStoreBase,
        embedder: EmbedderBase,
        reranker: RerankerBase | None = None,
    ):
        self.vector_store = vector_store
        self.sparse_store = sparse_store
        self.embedder = embedder
        self.reranker = reranker or FlashRankReranker()
        self._use_sql_hybrid = hasattr(vector_store, "hybrid_search")
        logger.info(
            f"HybridRetriever ready — "
            f"{'SQL hybrid (Supabase)' if self._use_sql_hybrid else 'Python RRF (Milvus+BM25S)'}"
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval.
        Returns top_k reranked chunks ready for LLM context.
        """
        top_k = top_k or config.retrieval.top_k_rerank

        if self._use_sql_hybrid:
            fused = self._retrieve_sql(query)
        else:
            fused = self._retrieve_python(query)

        # FlashRank reranking — same for both paths
        reranked = self.reranker.rerank(query, fused, top_k=top_k)
        logger.info(f"HybridRetriever → {len(reranked)} final chunks for: '{query[:60]}'")
        return reranked

    def _retrieve_sql(self, query: str) -> list[RetrievedChunk]:
        """
        Path A: single SQL query handles dense + sparse + RRF.
        Supabase round trips: 1 (embed) + 1 (hybrid_search) = 2 total.
        """
        query_embedding = self.embedder.embed_query(query)

        # Fetch more candidates than top_k_rerank so FlashRank has room to work
        candidate_k = max(config.retrieval.top_k_dense, config.retrieval.top_k_sparse)

        fused = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=candidate_k,
            rrf_k=config.retrieval.rrf_k,
        )
        logger.debug(f"SQL hybrid: {len(fused)} fused candidates")
        return fused

    def _retrieve_python(self, query: str) -> list[RetrievedChunk]:
        """
        Path B: separate dense + sparse calls, fused in Python via RRF.
        Used for Milvus + BM25S.
        """
        query_embedding = self.embedder.embed_query(query)

        dense_results = self.vector_store.search(
            query_embedding, top_k=config.retrieval.top_k_dense
        )
        logger.debug(f"Dense: {len(dense_results)} results")

        if hasattr(self.sparse_store, "search_sparse"):
            sparse_results = self.sparse_store.search_sparse(
                query, top_k=config.retrieval.top_k_sparse
            )
        else:
            sparse_results = self.sparse_store.search(
                query, top_k=config.retrieval.top_k_sparse
            )
        logger.debug(f"Sparse: {len(sparse_results)} results")

        fused = reciprocal_rank_fusion(
            dense_results, sparse_results, k=config.retrieval.rrf_k
        )
        logger.debug(f"After Python RRF: {len(fused)} unique chunks")
        return fused
