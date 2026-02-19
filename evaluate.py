"""
Evaluation entrypoint — build golden set, run RAGAS, analyze failures.

Usage:
  python evaluate.py build          # Generate golden set from ingested docs
  python evaluate.py run            # Run RAGAS over saved golden set
  python evaluate.py analyze        # Analyze latest results for fix recommendations
  python evaluate.py all            # build → run → analyze in sequence
  python evaluate.py run --samples 10   # Quick run with subset
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


def cmd_build(n_samples: int = 25):
    """Generate and save a golden test set."""
    from core.llm_client import OllamaClient
    from evaluation.golden_set import GoldenSetBuilder

    print(f"\n📋 Building golden set ({n_samples} samples)...")
    builder = GoldenSetBuilder(llm=OllamaClient())
    samples = builder.generate_from_chunks(n_samples=n_samples)
    builder.save(samples)
    builder.export_csv(samples)

    print(f"\n✅ Golden set saved:")
    print(f"   JSON → ./evaluation/golden_set.json  (used by evaluator)")
    print(f"   CSV  → ./evaluation/golden_set.csv   (review & edit this)")
    print(f"\n⚠️  IMPORTANT: Review golden_set.csv before running evaluation.")
    print(f"   Auto-generated Q&A pairs may have errors — human review improves eval quality.\n")


def cmd_run(n_samples: int | None = None):
    """Run RAGAS evaluation over the golden set."""
    from evaluation.golden_set import GoldenSetBuilder, GoldenSample
    from evaluation.ragas_evaluator import RagasEvaluator
    from agents.graph import build_rag_graph
    from core.llm_client import OllamaClient

    # Load golden set
    builder = GoldenSetBuilder(llm=OllamaClient())
    try:
        samples = builder.load_from_file()
    except FileNotFoundError:
        print("❌ Golden set not found. Run: python evaluate.py build")
        sys.exit(1)

    if n_samples:
        import random
        samples = random.sample(samples, min(n_samples, len(samples)))
        print(f"  Using {len(samples)} samples (subset)")

    print(f"\n🔬 Running RAGAS evaluation on {len(samples)} samples...")
    print(f"   (This will make {len(samples) * 4} LLM calls — may take a few minutes)\n")

    app = build_rag_graph()
    evaluator = RagasEvaluator()
    result = evaluator.run(samples=samples, app=app)

    return result


def cmd_analyze():
    """Analyze the most recent evaluation results."""
    from evaluation.failure_analyzer import FailureAnalyzer

    print("\n🔍 Analyzing latest evaluation results...\n")
    analyzer = FailureAnalyzer()
    report = analyzer.analyze_latest()
    print(report)


def main():
    commands = {
        "build":   cmd_build,
        "run":     cmd_run,
        "analyze": cmd_analyze,
    }

    args = sys.argv[1:]
    if not args or args[0] not in (*commands, "all"):
        print(__doc__)
        sys.exit(0)

    cmd = args[0]

    # Parse --samples flag
    n_samples = None
    if "--samples" in args:
        idx = args.index("--samples")
        try:
            n_samples = int(args[idx + 1])
        except (IndexError, ValueError):
            print("⚠️  --samples requires an integer value")
            sys.exit(1)

    if cmd == "all":
        cmd_build(n_samples or 25)
        print("\n" + "─" * 60 + "\n")
        result = cmd_run(n_samples)
        print("\n" + "─" * 60 + "\n")
        cmd_analyze()
    elif cmd == "build":
        cmd_build(n_samples or 25)
    elif cmd == "run":
        cmd_run(n_samples)
    elif cmd == "analyze":
        cmd_analyze()


if __name__ == "__main__":
    main()
