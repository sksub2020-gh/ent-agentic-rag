"""
Ingestion pipeline — config-driven backend via store_factory.
Supports Supabase, Qdrant, and Milvus+BM25S via STORE_BACKEND in .env

Usage:
    python cli/ingestion_pipeline.py ./data/raw/docling.pdf   # ingest file
    python cli/ingestion_pipeline.py ./data/raw/              # ingest directory
    python cli/ingestion_pipeline.py inspect                  # show store stats
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
DIVIDER = "=" * 55
SUBDIV  = "-" * 55


def ingest(path: str) -> None:
    from ingestion.docling_chunker import DoclingHybridChunker
    from ingestion.embedder import MpetEmbedder
    from retrieval.store_factory import build_stores

    embedder = MpetEmbedder()
    chunker  = DoclingHybridChunker(embedder.tokenizer)
    
    vector_store, sparse_store = build_stores(embedder)

    # Collect files
    p = Path(path)
    files = (
        [f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if p.is_dir() else [p]
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

    # Explicitly close store — flushes Qdrant local SQLite to disk
    if hasattr(vector_store, "close"):
        vector_store.close()
        logger.info("Store closed and flushed to disk")


def inspect() -> None:
    """Print vector store stats — collection info, point count, sample chunks."""
    from ingestion.embedder import MpetEmbedder
    from retrieval.store_factory import build_stores
    from config.settings import config

    embedder = MpetEmbedder()
    vector_store, _ = build_stores(embedder)

    print("")
    print(DIVIDER)
    print("  Vector Store Inspection")
    print(SUBDIV)
    print(f"  Backend:    {config.store_backend}")
    print(f"  Model:      {config.embedding.model_name}")

    total = vector_store.count()
    print(f"  Points:     {total}")

    # Qdrant-specific collection info
    if hasattr(vector_store, "client"):
        try:
            info   = vector_store.client.get_collection(vector_store.collection)
            dim    = info.config.params.vectors.get("dense").size
            status = info.status
            print(f"  Dimension:  {dim}")
            print(f"  Collection: {vector_store.collection}")
            print(f"  Status:     {status}")
        except Exception as e:
            print(f"  Collection info error: {e}")

    # Sample 3 chunks — verify payload and vectors are populated
    print(SUBDIV)
    print("  Sample chunks (first 3):")
    try:
        if hasattr(vector_store, "client"):
            points, _ = vector_store.client.scroll(
                collection_name=vector_store.collection,
                limit=3,
                with_payload=True,
                with_vectors=True,
            )
            for i, p in enumerate(points, 1):
                content_val  = str(p.payload.get("content", "EMPTY"))[:60] if p.payload else "EMPTY"
                vec          = p.vector or {}
                dense_len    = len(vec.get("dense", []))
                sparse_obj   = vec.get("sparse", {})
                sparse_len   = len(sparse_obj.get("values", [])) if isinstance(sparse_obj, dict) else 0
                page         = p.payload.get("page", "?") if p.payload else "?"
                section      = str(p.payload.get("section", ""))[:30] if p.payload else ""
                print(f"\n  [{i}] page={page} | {section}")
                print(f"       content: '{content_val}...'")
                print(f"       dense={dense_len}d  sparse_nnz={sparse_len}")
        else:
            print("  (detailed inspection only available for Qdrant backend)")
    except Exception as e:
        print(f"  Sample read error: {e}")

    if hasattr(vector_store, "close"):
        vector_store.close()

    print(DIVIDER)
    print("")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "inspect":
        inspect()
    else:
        ingest(sys.argv[1])
