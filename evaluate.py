"""
Evaluation entrypoint.

Usage:
  python cli/evaluate.py run                        # agentic mode (default)
  python cli/evaluate.py run --mode linear          # linear RAG mode
  python cli/evaluate.py run --samples 10           # subset of golden set
  python cli/evaluate.py run --mode linear --samples 10
  python cli/evaluate.py analyze                    # analyze latest results

Cache behaviour:
  Pipeline outputs are cached per mode in evaluation/results/pipeline_cache_{mode}.json
  Re-running reuses cache — delete cache file to force fresh pipeline run.
"""
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_run(n_samples: int | None = None, mode: str = "agentic"):
    from evaluation.golden_set import load_golden_set, summarise
    from evaluation.ragas_evaluator import RagasEvaluator
    from agents.graph import build_rag_graph

    if mode not in ("agentic", "linear"):
        print(f"❌ Unknown mode: '{mode}'. Choose from: agentic, linear")
        sys.exit(1)

    try:
        samples = load_golden_set(exclude_types=["Irrelevant"])
    except FileNotFoundError:
        print("❌ Golden set not found at evaluation/golden_set.json")
        sys.exit(1)

    summarise(samples)

    if n_samples:
        import random
        samples = random.sample(samples, min(n_samples, len(samples)))
        print(f"  Using {len(samples)} samples (subset)\n")

    print(f"🔬 Running RAGAS [{mode}] on {len(samples)} samples...")
    print(f"   (~{len(samples) * 4} LLM calls — may take several minutes)\n")

    # Build retriever and LLM once — shared between graph and linear mode
    # Prevents two QdrantClient instances opening the same local file simultaneously
    from retrieval.store_factory import build_retriever
    from core.llm_client import LLMClient
    retriever = build_retriever()
    llm       = LLMClient()
    app       = build_rag_graph(llm=llm, retriever=retriever)

    evaluator = RagasEvaluator()
    evaluator.run(samples=samples, app=app, mode=mode, retriever=retriever, llm=llm)


def cmd_analyze():
    from evaluation.failure_analyzer import FailureAnalyzer
    print("\n🔍 Analyzing latest evaluation results...\n")
    analyzer = FailureAnalyzer()
    print(analyzer.analyze_latest())


def _parse_args():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    n_samples = None
    mode = "agentic"

    if "--samples" in args:
        idx = args.index("--samples")
        try:
            n_samples = int(args[idx + 1])
        except (IndexError, ValueError):
            print("⚠️  --samples requires an integer value")
            sys.exit(1)

    if "--mode" in args:
        idx = args.index("--mode")
        try:
            mode = args[idx + 1]
        except IndexError:
            print("⚠️  --mode requires a value: agentic or linear")
            sys.exit(1)

    return cmd, n_samples, mode


def main():
    cmd, n_samples, mode = _parse_args()

    if cmd == "run":
        cmd_run(n_samples=n_samples, mode=mode)
    elif cmd == "analyze":
        cmd_analyze()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
