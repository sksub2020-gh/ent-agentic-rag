"""
Qdrant store — local dev backend replacing Milvus-Lite + BM25S.

Dense search:  Qdrant vectors         — cosine similarity
Sparse search: Qdrant sparse (BM42)   — FastEmbed neural sparse
Hybrid fusion: Qdrant native RRF      — single query_points() call
Scalar filter: Qdrant payload filters — page, section, doc_type, version

Deployment modes (QDRANT_MODE in .env):
    local  → file-based, no server  (default)
    memory → in-memory, testing
    remote → self-hosted or Qdrant Cloud

Add to requirements.txt:
    qdrant-client>=1.7.0
    fastembed>=0.2.0
"""
import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from core.interfaces import VectorStoreBase, SparseStoreBase, Chunk, RetrievedChunk
from config.settings import config

logger = logging.getLogger(__name__)

DENSE_VECTOR  = "dense"
SPARSE_VECTOR = "sparse"


class QdrantStore(VectorStoreBase, SparseStoreBase):
    """
    Single Qdrant store — dense + sparse + hybrid in one.
    Same dual-interface pattern as SupabaseStore.

    HybridRetriever detects hybrid_search() via hasattr() → Path A.
    FlashRank reranks fused candidates on top.
    """

    def __init__(self, dimension: int | None = None):
        self.collection = config.qdrant.collection_name
        self.dimension  = dimension or config.qdrant.dimension
        mode = config.qdrant.mode.lower()

        if mode == "memory":
            self.client = QdrantClient(":memory:")
            logger.info("QdrantStore → in-memory")
        elif mode == "local":
            self.client = QdrantClient(path=config.qdrant.path)
            logger.info(f"QdrantStore → local: {config.qdrant.path}")
        else:
            self.client = QdrantClient(
                url=config.qdrant.url,
                api_key=config.qdrant.api_key.get_secret_value() or None,
            )
            logger.info(f"QdrantStore → remote: {config.qdrant.url}")

    # ── Collection ────────────────────────────────────────────────────────────

    def _ensure_collection(self, dimension: int) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

        # Payload indexes for scalar filtering
        for field in ("page", "section", "doc_type", "version", "doc_id"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        logger.info(f"Collection '{self.collection}' created — {dimension}d dense + sparse")

    # ── VectorStoreBase ───────────────────────────────────────────────────────

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        if chunks[0].embedding:
            self._ensure_collection(len(chunks[0].embedding))

        points = []
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.chunk_id} missing embedding — skipping")
                continue

            meta = chunk.metadata
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id":   chunk.doc_id,
                "content":  chunk.content,
                "page":     str(meta.get("page", "")),
                "section":  str(meta.get("section", "")),
                "doc_type": str(meta.get("doc_type", "")),
                "version":  str(meta.get("version", "")),
                "source":   str(meta.get("source", "")),
            }

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(models.PointStruct(
                id=point_id,
                vector={DENSE_VECTOR: chunk.embedding},
                payload=payload,
            ))

        if points:
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info(f"Upserted {len(points)} dense vectors")

        self._upsert_sparse(chunks)

    def _get_sparse_encoder(self):
        """
        Lazy-load FastEmbed sparse encoder.
        Cached on instance to avoid reloading model on every call.
        """
        if not hasattr(self, "_sparse_encoder"):
            try:
                from fastembed import SparseTextEmbedding
                self._sparse_encoder = SparseTextEmbedding(
                    model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
                )
                logger.info("FastEmbed BM42 sparse encoder loaded")
            except ImportError:
                raise ImportError(
                    "fastembed is required for Qdrant sparse search. "
                    "Install it: pip install fastembed"
                )
        return self._sparse_encoder

    def _encode_sparse(self, texts: list[str]):
        """Encode texts to sparse vectors using FastEmbed BM42."""
        encoder = self._get_sparse_encoder()
        return list(encoder.embed(texts))

    def _upsert_sparse(self, chunks: list[Chunk]) -> None:
        """Compute BM42 sparse vectors via FastEmbed and upsert."""
        try:
            texts = [c.content for c in chunks if c.embedding]
            ids   = [
                str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id))
                for c in chunks if c.embedding
            ]

            sparse_embeddings = self._encode_sparse(texts)

            points = [
                models.PointStruct(
                    id=point_id,
                    vector={
                        SPARSE_VECTOR: models.SparseVector(
                            indices=se.indices.tolist(),
                            values=se.values.tolist(),
                        )
                    },
                    payload={},
                )
                for point_id, se in zip(ids, sparse_embeddings)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            logger.debug(f"Upserted {len(points)} sparse vectors")

        except Exception as e:
            logger.warning(f"Sparse upsert failed ({e})")

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Dense-only search — used by Python RRF path."""
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,           # modern API — just pass the vector
            using=DENSE_VECTOR,
            limit=top_k,
            query_filter=self._build_filter(filter),
            with_payload=True,
        ).points
        return [self._hit_to_retrieved(r, "dense") for r in results]

    def delete(self, chunk_ids: list[str]) -> None:
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, cid)) for cid in chunk_ids]
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=point_ids),
        )
        logger.info(f"Deleted {len(chunk_ids)} chunks")

    def count(self) -> int:
        try:
            return self.client.count(collection_name=self.collection).count
        except Exception:
            return 0

    # ── SparseStoreBase ───────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        """No-op — sparse vectors upserted during upsert()."""
        logger.info("QdrantStore.index() — sparse handled during upsert, no action needed")

    def search_sparse(
        self,
        query: str,
        top_k: int,
        filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Sparse-only search — used by Python RRF path."""
        try:
            se = self._encode_sparse([query])[0]
            results = self.client.query_points(
                collection_name=self.collection,
                query=models.SparseVector(
                    indices=se.indices.tolist(),
                    values=se.values.tolist(),
                ),
                using=SPARSE_VECTOR,
                limit=top_k,
                query_filter=self._build_filter(filter),
                with_payload=True,
            ).points
            return [self._hit_to_retrieved(r, "sparse") for r in results]
        except Exception as e:
            logger.warning(f"Sparse search failed ({e}) — returning empty")
            return []

    # ── Native Hybrid Search (Path A) ─────────────────────────────────────────

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int,
        rrf_k: int = 60,
        filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Native Qdrant hybrid — dense + sparse + RRF in one query_points() call.
        Detected by HybridRetriever via hasattr() → Path A.
        """
        qdrant_filter = self._build_filter(filter)

        try:
            se = self._encode_sparse([query_text])[0]

            results = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=query_embedding,
                        using=DENSE_VECTOR,
                        limit=top_k * 2,
                        filter=qdrant_filter,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=se.indices.tolist(),
                            values=se.values.tolist(),
                        ),
                        using=SPARSE_VECTOR,
                        limit=top_k * 2,
                        filter=qdrant_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            ).points

            return [self._hit_to_retrieved(r, "hybrid") for r in results]

        except Exception as e:
            logger.warning(f"Hybrid search failed ({e}) — falling back to dense")
            return self.search(query_embedding, top_k, filter)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_filter(self, filter: dict | None) -> models.Filter | None:
        if not filter:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v)
                )
                for k, v in filter.items()
            ]
        )

    def _hit_to_retrieved(self, hit: Any, source: str) -> RetrievedChunk:
        payload = hit.payload or {}
        chunk = Chunk(
            chunk_id=payload.get("chunk_id", str(hit.id)),
            doc_id=payload.get("doc_id", ""),
            content=payload.get("content", ""),
            metadata={
                "source":   payload.get("source", ""),
                "page":     payload.get("page", ""),
                "section":  payload.get("section", ""),
                "doc_type": payload.get("doc_type", ""),
                "version":  payload.get("version", ""),
            },
        )
        return RetrievedChunk(chunk=chunk, score=float(hit.score), source=source)
