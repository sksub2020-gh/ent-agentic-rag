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
    from agents.rag_node import RAG_SYSTEM_PROMPT, build_context_prompt
    from retrieval.store_factory import build_retriever

    embedder  = MpetEmbedder()
    llm       = LLMClient()

    retriever = build_retriever(embedder=embedder)

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

    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # 1. Initialize Phoenix and the OpenTelemetry bridge
    px.launch_app()
    register(project_name=config.project_name, auto_instrument=True)
    # 2. Prevent the "Already instrumented" warning
    if not LangChainInstrumentor().is_instrumented_by_opentelemetry:
        LangChainInstrumentor().instrument()

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
