"""
RAG query CLI entrypoint.
Usage: python cli/rag_query.py "Your question here"
"""
import logging
import sys
from config.settings import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def query(question: str) -> dict:
    from core.llm_client import LLMClient
    from ingestion.embedder import MpetEmbedder
    from retrieval.hybrid_retriever import HybridRetriever, FlashRankReranker
    from agents.rag_node import RAG_SYSTEM_PROMPT, build_context_prompt

    embedder  = MpetEmbedder()
    llm       = LLMClient()

    if config.store_backend == "supabase":
        from retrieval.supabase_store import SupabaseStore
        store = SupabaseStore()
        vector_store = sparse_store = store
    elif config.store_backend == "milvus":
        from retrieval.milvus_store import MilvusLiteStore
        from retrieval.bm25_store import BM25SStore
        vector_store = MilvusLiteStore(dimension=embedder.dimension)
        sparse_store = BM25SStore()
    else:
        print(f"⚠️  Vector Backend {config.store_backend} is not implemented.")
        sys.exit(1)
    
    retriever = HybridRetriever(
        vector_store=vector_store,
        sparse_store=sparse_store,
        embedder=embedder,
        reranker=FlashRankReranker(),
    )

    chunks = retriever.retrieve(question)
    if not chunks:
        return {"answer": "No relevant context found.", "chunks": [], "query": question}

    answer = llm.generate(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=build_context_prompt(question, chunks),
    )
    return {
        "query":  question,
        "answer": answer,
        "chunks": [
            {
                "content": rc.chunk.content[:200] + "...",
                "score":   round(rc.score, 4),
                "source":  rc.chunk.metadata.get("source", ""),
                "page":    rc.chunk.metadata.get("page", ""),
            }
            for rc in chunks
        ],
    }


if __name__ == "__main__":
    from core.llm_client import LLMClient

    llm = LLMClient()
    if not llm.health_check():
        print("⚠️  Ollama not reachable. Run: ollama serve")
        sys.exit(1)

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    print(f"\n🔍 Query: {question}\n")

    result = query(question)

    print(f"💬 Answer:\n{result['answer']}\n")
    print(f"📚 Sources ({len(result['chunks'])} chunks):")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"  [{i}] score={chunk['score']} | {chunk['source']} p.{chunk['page']}")
        print(f"       {chunk['content'][:100]}...")
