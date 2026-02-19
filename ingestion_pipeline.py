"""
Main ingestion pipeline — ties Docling + mpeT + Milvus + BM25S together.
Run this to ingest documents before querying.
"""
import logging
from pathlib import Path

from ingestion.docling_chunker import DoclingHybridChunker
from ingestion.embedder import MpetEmbedder
from retrieval.milvus_store import MilvusLiteStore
from retrieval.bm25_store import BM25SStore
from core.interfaces import Chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow:
    Document → Docling chunks → mpeT embeddings → Milvus (dense) + BM25S (sparse)
    """

    def __init__(self):
        logger.info("Initializing ingestion pipeline...")
        self.embedder = MpetEmbedder()
        self.chunker = DoclingHybridChunker(self.embedder.tokenizer)
        self.vector_store = MilvusLiteStore(self.embedder.dimension)
        self.sparse_store = BM25SStore()

    def ingest_file(self, file_path: str) -> list[Chunk]:
        """Ingest a single file (PDF, HTML, DOCX)."""
        logger.info(f"Ingesting: {file_path}")

        # 1. Chunk
        chunks = self.chunker.chunk_from_file(file_path)
        if not chunks:
            logger.warning(f"No chunks produced from {file_path}")
            return []

        # 2. Embed (batch for efficiency)
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 3. Store dense (Milvus)
        self.vector_store.upsert(chunks)

        # 4. Store sparse (BM25S) — rebuilds index including new chunks
        # For production at scale, collect all chunks then call index() once
        all_chunks = self._get_all_chunks_for_bm25(chunks)
        self.sparse_store.index(all_chunks)

        logger.info(f"✓ Ingested {len(chunks)} chunks from {Path(file_path).name}")
        return chunks

    def ingest_directory(self, dir_path: str, extensions: list[str] | None = None) -> int:
        """Ingest all supported files from a directory."""
        extensions = extensions or [".pdf", ".html", ".htm"]
        dir_path = Path(dir_path)
        files = [f for f in dir_path.rglob("*") if f.suffix.lower() in extensions]

        logger.info(f"Found {len(files)} files in {dir_path}")

        all_chunks = []
        for file in files:
            try:
                chunks = self.ingest_file(str(file))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to ingest {file}: {e}")

        # Rebuild BM25 once for all docs (more efficient than per-file rebuild)
        if all_chunks:
            self.sparse_store.index(all_chunks)

        total = self.vector_store.count()
        logger.info(f"Ingestion complete — total chunks in store: {total}")
        return total

    def _get_all_chunks_for_bm25(self, new_chunks: list[Chunk]) -> list[Chunk]:
        """
        BM25S needs all chunks to rebuild index.
        In Phase 1, we combine new + existing from the BM25 store's loaded corpus.
        Production note: For large-scale incremental ingestion, switch to Elasticsearch.
        """
        existing = self.sparse_store._corpus_chunks or []
        existing_ids = {c.chunk_id for c in existing}
        combined = list(existing)
        for chunk in new_chunks:
            if chunk.chunk_id not in existing_ids:
                combined.append(chunk)
        return combined


# ---------------------------------------------------------------------------
# Quick test / demo usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # if len(sys.argv) < 2:
    #     print("Usage: python ingestion_pipeline.py <path_to_file_or_dir>")
    #     print("Example: python ingestion_pipeline.py ./docs/sample.pdf")
    #     sys.exit(1)

    target = "./data/raw/docling.pdf"#sys.argv[1]
    pipeline = IngestionPipeline()

    path = Path(target)
    if path.is_dir():
        total = pipeline.ingest_directory(target)
        print(f"\n✅ Done — {total} total chunks indexed")
    elif path.is_file():
        chunks = pipeline.ingest_file(target)
        print(f"\n✅ Done — {len(chunks)} chunks indexed from {path.name}")
    else:
        print(f"Error: {target} is not a valid file or directory")
        sys.exit(1)
