"""
Basic RAG query pipeline — Phase 1 smoke test.
Dense retrieval only (no BM25 yet) to validate the core loop.
Run after ingestion_pipeline.py.
"""
import logging

from core.llm_client import OllamaClient
from ingestion.embedder import MpetEmbedder
from retrieval.milvus_store import MilvusLiteStore
from retrieval.bm25_store import BM25SStore
from retrieval.hybrid_retriever import HybridRetriever, FlashRankReranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."
Always cite the source document and page number when available."""


def build_context_prompt(query: str, retrieved_chunks) -> str:
    """Format retrieved chunks into an LLM-ready prompt."""
    context_blocks = []
    for i, rc in enumerate(retrieved_chunks, 1):
        meta = rc.chunk.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        section = meta.get("section", "")
        header = f"[{i}] Source: {source} | Page: {page}"
        if section:
            header += f" | Section: {section}"
        context_blocks.append(f"{header}\n{rc.chunk.content}")

    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def query(question: str, use_hybrid: bool = True) -> dict:
    """
    Run a single RAG query.
    Returns dict with answer, chunks, and scores for inspection.
    """
    embedder = MpetEmbedder()
    vector_store = MilvusLiteStore()
    sparse_store = BM25SStore()
    llm = OllamaClient()
    reranker = FlashRankReranker()

    retriever = HybridRetriever(
        vector_store=vector_store,
        sparse_store=sparse_store,
        embedder=embedder,
        reranker=reranker,
    )

    # Retrieve
    chunks = retriever.retrieve(question)

    if not chunks:
        return {"answer": "No relevant context found.", "chunks": [], "query": question}

    # Build prompt and generate
    user_prompt = build_context_prompt(question, chunks)
    answer = llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

    return {
        "query": question,
        "answer": answer,
        "chunks": [
            {
                "content": rc.chunk.content[:200] + "...",
                "score": round(rc.score, 4),
                "source": rc.chunk.metadata.get("source", ""),
                "page": rc.chunk.metadata.get("page", ""),
            }
            for rc in chunks
        ],
    }


if __name__ == "__main__":
    import sys
    import json

    # Health check
    llm = OllamaClient()
    if not llm.health_check():
        print("⚠️  Ollama not reachable. Start it with: ollama serve")
        sys.exit(1)

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    print(f"\n🔍 Query: {question}\n")

    result = query(question)

    print(f"💬 Answer:\n{result['answer']}\n")
    print(f"📚 Sources used ({len(result['chunks'])} chunks):")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"  [{i}] score={chunk['score']} | {chunk['source']} p.{chunk['page']}")
        print(f"       {chunk['content'][:100]}...")
