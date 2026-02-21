"""
Ingestion pipeline — config-driven backend selection.
Supports Supabase, Qdrant, and Milvus+BM25S via STORE_BACKEND in .env

Usage:
    python cli/ingestion_pipeline.py ./data/raw/
    python cli/ingestion_pipeline.py ./data/raw/docling.pdf
"""

import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".docx"}


# ── Pipeline ──────────────────────────────────────────────────────────────────


def ingest(path: str) -> None:
    from ingestion.docling_chunker import DoclingHybridChunker
    from ingestion.embedder import MpetEmbedder
    from retrieval.store_factory import build_stores

    embedder = MpetEmbedder()
    chunker = DoclingHybridChunker(embedder.tokenizer)
    vector_store, sparse_store = build_stores(embedder)

    # Collect files
    p = Path(path)
    files = (
        [f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if p.is_dir()
        else [p]
    )

    if not files:
        logger.warning(f"No supported files found at: {path}")
        return

    logger.info(f"Ingesting {len(files)} file(s)...")

    all_chunks = []
    for file in files:
        logger.info(f"Processing: {file.name}")
        try:
            chunks = chunker.chunk_from_file(str(file))
            if not chunks:
                logger.warning(f"No chunks from {file.name}")
                continue

            # Embed
            embeddings = embedder.embed([c.content for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

            # Store dense
            vector_store.upsert(chunks)
            all_chunks.extend(chunks)
            logger.info(f"  ✓ {len(chunks)} chunks from {file.name}")

        except Exception as e:
            logger.error(f"  ✗ Failed {file.name}: {e}")

    # Sparse index — called once after all files for stores that need full corpus
    # (BM25S needs full rebuild; Supabase/Qdrant handle it during upsert — no-op)
    if all_chunks:
        sparse_store.index(all_chunks)

    logger.info(f"Ingestion complete — {len(all_chunks)} chunks indexed")
    logger.info(f"Total in store: {vector_store.count()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli/ingestion_pipeline.py <file_or_directory>")
        sys.exit(1)
    ingest(sys.argv[1])
