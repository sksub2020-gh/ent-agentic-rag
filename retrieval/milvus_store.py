"""
Milvus-Lite vector store — local, no server needed.
Implements VectorStoreBase → swap to Pinecone by writing PineconeStore(VectorStoreBase).
"""
import logging
from pymilvus import MilvusClient

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

    def __init__(self):
        import os
        os.makedirs("./data", exist_ok=True)

        self.client = MilvusClient(uri=config.milvus.uri)
        self.collection = config.milvus.collection_name
        self._ensure_collection()
        logger.info(f"MilvusLiteStore ready → {config.milvus.uri} | collection: {self.collection}")

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self.client.has_collection(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                dimension=config.embedding.dimension,
                metric_type=config.milvus.metric_type,
                id_type="string",
                max_length=128,          # chunk_id max length
            )
            logger.info(f"Collection '{self.collection}' created")
        else:
            logger.info(f"Collection '{self.collection}' already exists")

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert chunks. Chunks must have embeddings set before calling."""
        if not chunks:
            return

        import json
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
        import json
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
