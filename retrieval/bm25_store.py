"""
BM25S sparse index — persisted to disk.
Implements SparseStoreBase → swap to Elasticsearch by writing ElasticStore(SparseStoreBase).
"""
import json
import logging
import pickle
from pathlib import Path

import bm25s

from core.interfaces import SparseStoreBase, Chunk, RetrievedChunk
from config.settings import config

logger = logging.getLogger(__name__)


class BM25SStore(SparseStoreBase):
    """
    BM25S sparse full-text index.
    - 20-500x faster than rank_bm25
    - Saves/loads index to disk (survives restarts)
    - Pure NumPy, no server needed
    """

    def __init__(self):
        self.index_path = Path(config.bm25.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.retriever = None
        self._corpus_chunks: list[Chunk] = []   # needed to map results back to Chunk objects

        # Load existing index if available
        self._load_index()
        logger.info(f"BM25SStore ready → {self.index_path}")

    def index(self, chunks: list[Chunk]) -> None:
        """
        Build BM25S index from chunks.
        Call this after ingestion. Persists to disk automatically.
        Note: BM25S rebuilds the full index (append not supported natively).
        For large-scale incremental updates, use Elasticsearch instead.
        """
        if not chunks:
            return

        self._corpus_chunks = chunks
        corpus_texts = [chunk.content for chunk in chunks]

        # Tokenize corpus
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")

        # Build index
        self.retriever = bm25s.BM25(method=config.bm25.method)
        self.retriever.index(corpus_tokens)

        # Persist
        self._save_index()
        logger.info(f"BM25S indexed {len(chunks)} chunks")

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """BM25 keyword search."""
        if self.retriever is None or not self._corpus_chunks:
            logger.warning("BM25S index is empty — run index() first")
            return []

        query_tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = self.retriever.retrieve(query_tokens, k=min(top_k, len(self._corpus_chunks)))

        retrieved = []
        for idx, score in zip(results[0], scores[0]):
            chunk = self._corpus_chunks[idx]
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                score=float(score),
                source="sparse",
            ))

        return retrieved

    def _save_index(self):
        """Persist retriever + chunk mapping to disk."""
        self.retriever.save(str(self.index_path / "bm25s_index"))
        with open(self.index_path / "corpus_chunks.pkl", "wb") as f:
            pickle.dump(self._corpus_chunks, f)
        logger.info(f"BM25S index saved → {self.index_path}")

    def _load_index(self):
        """Load existing index from disk if present."""
        retriever_path = self.index_path / "bm25s_index"
        chunks_path = self.index_path / "corpus_chunks.pkl"

        if retriever_path.exists() and chunks_path.exists():
            try:
                self.retriever = bm25s.BM25.load(str(retriever_path))
                with open(chunks_path, "rb") as f:
                    self._corpus_chunks = pickle.load(f)
                logger.info(f"BM25S index loaded — {len(self._corpus_chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to load BM25S index: {e} — will rebuild on next ingest")
