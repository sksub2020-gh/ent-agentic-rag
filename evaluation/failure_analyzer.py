"""
Failure Analyzer — reads RAGAS results and gives actionable fix recommendations.

After running evaluation, this tells you exactly WHY scores are low and WHAT to change.
Maps each metric failure to specific pipeline components to tune.

Metric → Root Cause → Fix:
  faithfulness ↓      → LLM hallucinating beyond context   → stricter system prompt, smaller context window
  answer_relevancy ↓  → LLM off-topic / verbose            → prompt tuning, query rewriting
  context_precision ↓ → Retriever returning noisy chunks   → tune top_k, improve reranker threshold
  context_recall ↓    → Relevant chunks not being found    → improve chunking, add query expansion, tune BM25/dense weights
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_precision": 0.75,
    "context_recall":    0.70,
}

# Maps each failing metric to root causes and concrete fixes
FAILURE_PLAYBOOK = {
    "faithfulness": {
        "root_causes": [
            "LLM generating content beyond the provided context",
            "System prompt not strict enough about grounding",
            "Context window too large — LLM relies on parametric memory",
        ],
        "fixes": [
            "Tighten RAG_SYSTEM_PROMPT in rag_node.py — add 'Answer ONLY from the context below.'",
            "Reduce top_k_rerank in config/settings.py (try 3 instead of 5) — less noise for LLM",
            "Lower Critique node MAX_RETRIES threshold — be stricter about grounding check",
            "Add negative examples to system prompt: 'Never say X if not in context'",
        ],
        "langsmith_query": "filter runs where grounded=False and retry_count=0",
    },
    "answer_relevancy": {
        "root_causes": [
            "LLM answer is verbose or drifts from the original question",
            "Query is ambiguous — LLM answers a different interpretation",
            "Direct-route answers (no retrieval) are off-topic",
        ],
        "fixes": [
            "Add 'Be concise. Answer the exact question asked.' to RAG_SYSTEM_PROMPT",
            "Add query rewriting step before retrieval in rag_node.py",
            "Review router_node.py — check if 'direct' route is over-triggered",
            "Add few-shot examples of good answers to system prompt",
        ],
        "langsmith_query": "filter runs where route=direct — check if answers are relevant",
    },
    "context_precision": {
        "root_causes": [
            "Retriever returning too many irrelevant chunks",
            "Reranker not filtering noise effectively",
            "Chunk size too large — chunks contain mixed topics",
        ],
        "fixes": [
            "Reduce top_k_dense and top_k_sparse in config/settings.py",
            "Increase FlashRank score threshold — only keep chunks above 0.3",
            "Reduce max_tokens in DoclingConfig — smaller, more focused chunks",
            "Switch reranker model: try 'ms-marco-MultiBERT-L-12' for better precision",
        ],
        "langsmith_query": "inspect context field in RAG node — count irrelevant passages",
    },
    "context_recall": {
        "root_causes": [
            "Relevant chunks not found during retrieval",
            "Poor BM25/dense balance — hybrid weights off",
            "Chunking splits answers across chunk boundaries",
            "Query doesn't match chunk vocabulary (semantic gap)",
        ],
        "fixes": [
            "Increase top_k_dense and top_k_sparse in config/settings.py",
            "Add query expansion in rag_node.py — generate 3 query variants, retrieve for each",
            "Increase overlap_tokens in DoclingConfig — reduce boundary splitting",
            "Add HyDE (Hypothetical Document Embeddings) — embed a hypothetical answer, use for retrieval",
            "Review BM25S tokenization — ensure stopwords list isn't too aggressive",
        ],
        "langsmith_query": "compare retrieved chunks vs ground_truth_contexts — find missing chunks",
    },
}


class FailureAnalyzer:
    """
    Reads an EvalResult (or results JSON) and produces a prioritized fix plan.
    """

    def analyze(self, scores: dict[str, float], per_sample: list[dict] | None = None) -> str:
        """
        Analyze scores and return a formatted fix report.

        Args:
            scores:     {metric_name: score} dict from EvalResult
            per_sample: Optional list of per-sample results for deeper analysis
        """
        failing = {m: s for m, s in scores.items() if s < THRESHOLDS[m]}
        passing = {m: s for m, s in scores.items() if s >= THRESHOLDS[m]}

        lines = ["\n" + "═" * 60, "  FAILURE ANALYSIS & FIX RECOMMENDATIONS", "═" * 60]

        if not failing:
            lines.append("\n  ✅ All metrics passing! Consider tightening thresholds.")
            return "\n".join(lines)

        # Prioritize by how far below threshold
        prioritized = sorted(failing.items(), key=lambda x: THRESHOLDS[x[0]] - x[1], reverse=True)

        lines.append(f"\n  Passing:  {', '.join(f'{m}={s:.3f}' for m, s in passing.items()) or 'none'}")
        lines.append(f"  Failing:  {', '.join(f'{m}={s:.3f}' for m, s in prioritized)}")
        lines.append(f"\n  Fix priority order (worst first):\n")

        for rank, (metric, score) in enumerate(prioritized, 1):
            gap = THRESHOLDS[metric] - score
            playbook = FAILURE_PLAYBOOK[metric]

            lines.append(f"  {'─'*56}")
            lines.append(f"  [{rank}] {metric.upper()} — score={score:.3f}  gap={gap:.3f}")
            lines.append(f"\n  Root causes:")
            for cause in playbook["root_causes"]:
                lines.append(f"    • {cause}")
            lines.append(f"\n  Fixes (try in order):")
            for fix in playbook["fixes"]:
                lines.append(f"    → {fix}")
            lines.append(f"\n  LangSmith: {playbook['langsmith_query']}")
            lines.append("")

        # Per-sample patterns
        if per_sample:
            lines.append(f"  {'─'*56}")
            lines.append(f"  PER-SAMPLE PATTERNS:")
            lines.extend(self._per_sample_insights(per_sample))

        lines.append("═" * 60 + "\n")
        return "\n".join(lines)

    def analyze_from_file(self, path: str) -> str:
        """Load a saved eval JSON and analyze it."""
        with open(path) as f:
            data = json.load(f)
        return self.analyze(
            scores=data["scores"],
            per_sample=data.get("per_sample", []),
        )

    def analyze_latest(self, results_dir: str = "./evaluation/results") -> str:
        """Analyze the most recent evaluation result."""
        results = sorted(Path(results_dir).glob("eval_*.json"))
        if not results:
            return "No evaluation results found. Run evaluate.py first."
        return self.analyze_from_file(str(results[-1]))

    def _per_sample_insights(self, per_sample: list[dict]) -> list[str]:
        """Find patterns in per-sample failures."""
        lines = []

        # Retry rate
        retried = [s for s in per_sample if s.get("retry_count", 0) > 0]
        if retried:
            lines.append(f"    • {len(retried)}/{len(per_sample)} queries required retries "
                         f"— critique node is catching issues but retrieval still failing")

        # Direct route rate
        direct = [s for s in per_sample if s.get("route") == "direct"]
        if len(direct) > len(per_sample) * 0.3:
            lines.append(f"    • Router sent {len(direct)}/{len(per_sample)} queries to direct path "
                         f"— router may be over-confident, bypassing retrieval too often")

        # Question type breakdown
        by_type: dict[str, list] = {}
        for s in per_sample:
            qtype = s.get("question_type", "unknown")
            by_type.setdefault(qtype, []).append(s)

        for qtype, samples in by_type.items():
            ungrounded = [s for s in samples if not s.get("grounded", True)]
            if ungrounded:
                lines.append(f"    • {len(ungrounded)}/{len(samples)} '{qtype}' questions "
                             f"failed grounding — focus chunking/retrieval tuning here")

        return lines or ["    • No clear patterns detected — review LangSmith traces manually"]
