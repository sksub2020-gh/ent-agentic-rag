"""
Agentic RAG — main entrypoint.
Builds the graph once, then runs queries interactively or from CLI args.

Usage:
  python agentic_rag.py                          # interactive REPL
  python agentic_rag.py "What is X?"             # single query
  python agentic_rag.py --stream "What is X?"    # stream tokens (future)
"""
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def print_result(result: dict) -> None:
    """Pretty-print the agent result."""
    print("\n" + "═" * 60)
    print(f"  Query   : {result['query']}")
    print(f"  Route   : {result['route']} ({result['router_reasoning']})")
    print(f"  Retries : {result['retry_count']}")
    print(f"  Grounded: {'✅' if result['grounded'] else '❌'} — {result['critique_reasoning']}")
    print("─" * 60)
    print(f"\n{result['answer']}\n")

    if result["sources"]:
        print("📚 Sources:")
        for s in result["sources"]:
            print(f"  [{s['index']}] {s['source']} | Page {s['page']} | score={s['score']}")
    print("═" * 60 + "\n")


def main():
    from agents.graph import build_rag_graph, run_query
    from core.llm_client import OllamaClient

    # Health check before building the graph
    llm = OllamaClient()
    if not llm.health_check():
        print("⚠️  Ollama not reachable. Start it: ollama serve && ollama pull mistral")
        sys.exit(1)

    print("🔧 Building Agentic RAG graph...")
    app = build_rag_graph(llm=llm)
    print("✅ Graph ready\n")

    # ── Single query mode ────────────────────────────────────────────────────
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        query = " ".join(args)
        result = run_query(query, app=app)
        print_result(result)
        return

    # ── Interactive REPL ─────────────────────────────────────────────────────
    print("💬 Agentic RAG — type 'quit' to exit\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        try:
            result = run_query(query, app=app)
            print_result(result)
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            print(f"⚠️  Error: {e}\n")


if __name__ == "__main__":
    main()
