"""
Supabase ingestion pipeline — replaces ingestion_pipeline.py for Supabase deployments.

Usage:
    python ingestion_pipeline_supabase.py ./docs/sample.pdf
    python ingestion_pipeline_supabase.py ./docs/          # entire directory

Prerequisites:
    1. Run retrieval/supabase_migration.sql in Supabase SQL editor
    2. Set SUPABASE_CONNECTION_STRING in .env
    3. pip install psycopg2-binary pgvector
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".docx"}


def ingest(path: str) -> None:
    from ingestion.docling_chunker import DoclingHybridChunker
    from ingestion.embedder import MpetEmbedder
    from retrieval.supabase_store import SupabaseStore

    # Initialise components
    chunker  = DoclingHybridChunker()
    embedder = MpetEmbedder()
    store    = SupabaseStore()

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

    logger.info(f"Ingesting {len(files)} file(s) into Supabase...")

    total_chunks = 0
    for file in files:
        logger.info(f"Processing: {file}")
        try:
            # Chunk
            chunks = chunker.chunk_from_file(str(file))
            if not chunks:
                logger.warning(f"No chunks produced from {file}")
                continue

            # Embed
            texts = [c.content for c in chunks]
            embeddings = embedder.embed(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

            # Upsert to Supabase (dense + tsvector updated by trigger)
            store.upsert(chunks)
            total_chunks += len(chunks)
            logger.info(f"  ✓ {len(chunks)} chunks ingested from {file.name}")

        except Exception as e:
            logger.error(f"  ✗ Failed to process {file}: {e}")
            continue

    logger.info(f"Ingestion complete — {total_chunks} total chunks in Supabase")
    logger.info(f"Total chunks in store: {store.count()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingestion_pipeline_supabase.py <file_or_directory>")
        sys.exit(1)
    ingest(sys.argv[1])
