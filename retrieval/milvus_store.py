"""
Milvus-Lite vector store — local, no server needed.
Implements VectorStoreBase → swap to Pinecone by writing PineconeStore(VectorStoreBase).
"""
import logging
import json
from pymilvus import MilvusClient, DataType

from core.interfaces import VectorStoreBase, Chunk, RetrievedChunk
from config.settings import config

logger = logging.getLogger(__name__)

# Schema field names
FIELD_ID        = "chunk_id"
FIELD_DOC_ID    = "doc_id"
FIELD_CONTENT   = "content"
FIELD_EMBEDDING = "embedding"
FIELD_METADATA  = "metadata"      # stored as JSON string


class MilvusLiteStore(VectorStoreBase):
    """
    Milvus-Lite: stores everything in a local .db file.
    To swap → implement VectorStoreBase with PineconeStore, QdrantStore, etc.
    The rest of the pipeline never changes.
    """

    def __init__(self, dimension: int):
        import os
        os.makedirs("./data/index", exist_ok=True)

        self.client = MilvusClient(uri=config.milvus.uri)
        self.collection = config.milvus.collection_name
        self._ensure_collection(dimension)
        logger.info(f"MilvusLiteStore ready → {config.milvus.uri} | collection: {self.collection}")

    def _ensure_collection(self, dimension):
        """
        Create collection with explicit schema + vector index.

        Root cause of the original error: create_collection() shorthand always
        creates a field named 'id' regardless of id_type param. We must use
        the explicit schema API to define chunk_id as the primary key.
        """
        if self.client.has_collection(self.collection):
            logger.info(f"Collection '{self.collection}' already exists")
            return

        # ── Define schema explicitly ──────────────────────────────────────
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)

        schema.add_field(
            field_name=FIELD_ID,
            datatype=DataType.VARCHAR,
            max_length=128,
            is_primary=True,
        )
        schema.add_field(
            field_name=FIELD_DOC_ID,
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name=FIELD_CONTENT,
            datatype=DataType.VARCHAR,
            max_length=8192,
        )
        schema.add_field(
            field_name=FIELD_EMBEDDING,
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension,
        )
        schema.add_field(
            field_name=FIELD_METADATA,
            datatype=DataType.VARCHAR,
            max_length=2048,
        )

        # ── Define vector index ───────────────────────────────────────────
        # Milvus-Lite only supports: FLAT, IVF_FLAT, AUTOINDEX
        # AUTOINDEX = best available for the deployment (FLAT for lite, HNSW for full Milvus)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=FIELD_EMBEDDING,
            index_type="AUTOINDEX",
            metric_type=config.milvus.metric_type,
            params={"M": 16, "efConstruction": 256},
        )

        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Collection '{self.collection}' created with explicit schema + HNSW index")

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert chunks. Chunks must have embeddings set before calling."""
        if not chunks:
            return

        data = []
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.chunk_id} has no embedding — skipping")
                continue
            data.append({
                FIELD_ID:        chunk.chunk_id,
                FIELD_DOC_ID:    chunk.doc_id,
                FIELD_CONTENT:   chunk.content,
                FIELD_EMBEDDING: chunk.embedding,
                FIELD_METADATA:  json.dumps(chunk.metadata),
            })

        if data:
            self.client.upsert(collection_name=self.collection, data=data)
            logger.info(f"Upserted {len(data)} chunks into Milvus")

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Dense similarity search."""
        results = self.client.search(
            collection_name=self.collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=[FIELD_ID, FIELD_DOC_ID, FIELD_CONTENT, FIELD_METADATA],
        )

        retrieved = []
        for hit in results[0]:
            entity = hit["entity"]
            chunk = Chunk(
                chunk_id=entity[FIELD_ID],
                doc_id=entity[FIELD_DOC_ID],
                content=entity[FIELD_CONTENT],
                metadata=json.loads(entity.get(FIELD_METADATA, "{}")),
            )
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                score=hit["distance"],
                source="dense",
            ))

        return retrieved

    def delete(self, chunk_ids: list[str]) -> None:
        self.client.delete(
            collection_name=self.collection,
            ids=chunk_ids,
        )
        logger.info(f"Deleted {len(chunk_ids)} chunks from Milvus")

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))