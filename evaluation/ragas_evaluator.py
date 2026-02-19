"""
RAGAS Evaluator — measures RAG pipeline quality across 4 key metrics.

Metrics:
  faithfulness        — Is the answer supported by the retrieved context? (hallucination guard)
  answer_relevancy    — Does the answer actually address the question?
  context_precision   — Are the retrieved chunks relevant to the question?
  context_recall      — Did we retrieve all chunks needed to answer? (requires ground truth)

Each metric is 0.0–1.0. Enterprise targets:
  faithfulness      > 0.85  (critical — catches hallucinations)
  answer_relevancy  > 0.80
  context_precision > 0.75
  context_recall    > 0.70  (hardest to achieve)

RAGAS uses an LLM internally for evaluation. We point it at Ollama to stay fully local.
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from evaluation.golden_set import GoldenSample
from agents.graph import build_rag_graph, run_query
from config.settings import config

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./evaluation/results")

# Enterprise quality thresholds
THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_precision": 0.75,
    "context_recall":    0.70,
}


@dataclass
class EvalResult:
    """Results for a single evaluation run."""
    timestamp: str
    n_samples: int
    scores: dict[str, float]
    passed: dict[str, bool]         # Did each metric meet threshold?
    overall_pass: bool
    per_sample: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"\n{'═'*55}",
            f"  RAGAS Evaluation — {self.timestamp}",
            f"  Samples: {self.n_samples}",
            f"{'─'*55}",
        ]
        for metric, score in self.scores.items():
            threshold = THRESHOLDS.get(metric, 0.0)
            status = "✅" if self.passed.get(metric) else "❌"
            bar = _score_bar(score)
            lines.append(f"  {status} {metric:<22} {score:.3f}  {bar}  (threshold: {threshold})")
        lines.append(f"{'─'*55}")
        lines.append(f"  Overall: {'✅ PASS' if self.overall_pass else '❌ FAIL'}")
        lines.append(f"{'═'*55}\n")
        return "\n".join(lines)


def _score_bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


class RagasEvaluator:
    """
    Runs RAGAS evaluation against the full agentic RAG pipeline.
    Uses Ollama as the evaluation LLM to stay fully local.
    """

    def __init__(self):
        # Point RAGAS at Ollama via LangChain wrapper
        # RAGAS needs LangChain-compatible LLM/embedder interfaces
        eval_llm = ChatOpenAI(
            base_url=config.ollama.base_url,
            api_key=config.ollama.api_key,
            model=config.ollama.model,
            temperature=0.0,
        )
        eval_embedder = OpenAIEmbeddings(
            base_url=config.ollama.base_url,
            api_key=config.ollama.api_key,
            model="nomic-embed-text",   # Ollama embedding model for RAGAS internals
        )

        self.ragas_llm = LangchainLLMWrapper(eval_llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(eval_embedder)

        # Bind our LLM/embedder to each metric
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        for metric in self.metrics:
            metric.llm = self.ragas_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = self.ragas_embeddings

        logger.info("RagasEvaluator ready — using Ollama for evaluation LLM")

    def run(
        self,
        samples: list[GoldenSample],
        app=None,
        save_results: bool = True,
    ) -> EvalResult:
        """
        Run RAGAS evaluation over the golden set.

        Args:
            samples:      List of GoldenSample from golden_set.py
            app:          Pre-built LangGraph app (built if not provided)
            save_results: Persist results to JSON

        Returns:
            EvalResult with scores, pass/fail per metric, and per-sample breakdown
        """
        app = app or build_rag_graph()

        logger.info(f"Running RAGAS evaluation over {len(samples)} samples...")

        # ── Step 1: Run pipeline on each sample ───────────────────────────
        ragas_data = {
            "question":           [],
            "answer":             [],
            "contexts":           [],   # Retrieved chunk contents
            "ground_truth":       [],   # Expected answer (for context_recall)
        }
        per_sample_meta = []

        for i, sample in enumerate(samples):
            logger.info(f"  [{i+1}/{len(samples)}] {sample.question[:60]}")
            try:
                result = run_query(sample.question, app=app)

                # Pull retrieved context texts
                contexts = [
                    rc.chunk.content
                    for rc in result.get("retrieved_chunks_raw", [])
                ] or [""]  # RAGAS requires non-empty list

                ragas_data["question"].append(sample.question)
                ragas_data["answer"].append(result["answer"])
                ragas_data["contexts"].append(contexts)
                ragas_data["ground_truth"].append(sample.ground_truth_answer)

                per_sample_meta.append({
                    "question":       sample.question,
                    "answer":         result["answer"],
                    "ground_truth":   sample.ground_truth_answer,
                    "question_type":  sample.question_type,
                    "route":          result.get("route", ""),
                    "grounded":       result.get("grounded", False),
                    "retry_count":    result.get("retry_count", 0),
                    "n_contexts":     len(contexts),
                })

            except Exception as e:
                logger.error(f"  Pipeline failed for sample {i+1}: {e}")
                # Include failed sample with empty answer so RAGAS still scores it
                ragas_data["question"].append(sample.question)
                ragas_data["answer"].append("")
                ragas_data["contexts"].append([""])
                ragas_data["ground_truth"].append(sample.ground_truth_answer)
                per_sample_meta.append({"question": sample.question, "error": str(e)})

        # ── Step 2: Run RAGAS metrics ──────────────────────────────────────
        dataset = Dataset.from_dict(ragas_data)
        logger.info("Running RAGAS metrics (this may take a few minutes)...")

        ragas_result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
        )

        scores = {
            "faithfulness":      round(float(ragas_result["faithfulness"]), 4),
            "answer_relevancy":  round(float(ragas_result["answer_relevancy"]), 4),
            "context_precision": round(float(ragas_result["context_precision"]), 4),
            "context_recall":    round(float(ragas_result["context_recall"]), 4),
        }

        passed = {m: scores[m] >= THRESHOLDS[m] for m in scores}
        overall_pass = all(passed.values())

        result = EvalResult(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            n_samples=len(samples),
            scores=scores,
            passed=passed,
            overall_pass=overall_pass,
            per_sample=per_sample_meta,
        )

        print(result.summary())

        if save_results:
            self._save(result)

        return result

    def _save(self, result: EvalResult) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"eval_{ts}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info(f"Results saved → {path}")
