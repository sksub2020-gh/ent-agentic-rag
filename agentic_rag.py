"""
Agentic RAG — main entrypoint.
Builds the graph once, then runs queries interactively or from CLI args.

Usage:
  python cli/agentic_rag.py                      # interactive REPL (Read Eval Print Loop)
  python cli/agentic_rag.py "What is X?"         # single query
"""
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
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
    from core.llm_client import LLMClient
    from config.settings import config

    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # 1. Initialize Phoenix and the OpenTelemetry bridge
    px.launch_app()
    register(project_name=f"{config.project_name} - agentic-rag.py", auto_instrument=True)
    # 2. Prevent the "Already instrumented" warning
    if not LangChainInstrumentor().is_instrumented_by_opentelemetry:
        LangChainInstrumentor().instrument()

    # Health check before building the graph
    llm = LLMClient()
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
