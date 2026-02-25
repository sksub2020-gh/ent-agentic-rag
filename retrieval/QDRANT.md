```
MpetEmbedder          → dense vectors (your existing embedder, unchanged)
QdrantStore.upsert()  → stores dense (MpetEmbedder output) + sparse (BM42/FastEmbed)
QdrantStore.hybrid_search() → Qdrant RRF fuses both in one call
FlashRank             → reranks fused candidates
```

**fastembed is only used inside QdrantStore for sparse — your MpetEmbedder stays as the dense embedder, nothing changes there. They complement each other, not compete.**

```
pip install qdrant-client fastembed --break-system-packages
```
To ingest and test:
```
python cli/ingestion_pipeline.py ./data/raw/
python cli/rag_query.py "What is this document about?"
```