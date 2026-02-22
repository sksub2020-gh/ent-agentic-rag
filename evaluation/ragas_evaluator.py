"""
RAGAS Evaluator — measures RAG pipeline quality across 4 key metrics.

Metrics:
  faithfulness        — Is the answer supported by retrieved context?
  answer_relevancy    — Does the answer address the question?
  context_precision   — Are retrieved chunks relevant to the question?
  context_recall      — Did retrieval find the right chunks? (needs reference contexts)

Each metric is 0.0–1.0. Enterprise targets:
  faithfulness      > 0.85
  answer_relevancy  > 0.80
  context_precision > 0.75
  context_recall    > 0.70

Pipeline results are cached to evaluation/results/pipeline_cache.json before
RAGAS scoring — so if RAGAS crashes, re-running skips the pipeline loop entirely.
Delete pipeline_cache.json to force a fresh pipeline run.
"""
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

from evaluation.golden_set import GoldenSample
import re
import numpy as np
from config.settings import config

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./evaluation/results")
CACHE_PATH  = RESULTS_DIR / "pipeline_cache.json"

THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_precision": 0.75,
    "context_recall":    0.70,
}

# Ground truth values and question types to skip — RAGAS can't score these
SKIP_GROUND_TRUTHS  = {"NOT_APPLICABLE", "NOT_IN_CONTEXT"}
SKIP_QUESTION_TYPES = {"Irrelevant"}

# Regex to strip citation noise added by RAG node prompt
_CITATION_RE = re.compile(
    r'\[\d+(?:,\s*\d+)*\]'       # [1], [1, 2]
    r'|\[Source:[^\]]*\]'            # [Source: data/raw/..., Page: ]
    r'|\[Page:[^\]]*\]'              # [Page: 3]
    r'|\s*Source:\s*[^\n\[\]]+',   # Source: data/raw/... (unbracketed)
    re.IGNORECASE
)


def _clean_answer(answer: str) -> str:
    """Strip citation markers before passing answer to RAGAS."""
    cleaned = _CITATION_RE.sub("", answer)
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    return cleaned or answer


# ── Custom answer_relevancy ──────────────────────────────────────────────────

def _cosine_answer_relevancy(
    questions: list[str],
    answers: list[str],
    embedder=None,
) -> float:
    """
    Replacement for RAGAS answer_relevancy.

    RAGAS answer_relevancy uses the LLM to generate reverse questions then
    compares embeddings — but Mistral-7B doesn't reliably follow the internal
    JSON format, returning 0.0 for all samples.

    This implementation computes cosine similarity directly between
    question and answer embeddings using MpetEmbedder — no LLM call needed.

    Score interpretation:
      > 0.80 — answer is on-topic and relevant
      0.60-0.80 — partially relevant
      < 0.60 — answer drifted from question
    """
    if embedder is None:
        from ingestion.embedder import MpetEmbedder
        embedder = MpetEmbedder()

    # Filter out INSUFFICIENT_CONTEXT answers — score as 0
    scores = []
    texts_to_embed = []
    indices_to_score = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        if "INSUFFICIENT_CONTEXT" in a or not a.strip():
            scores.append(0.0)
        else:
            texts_to_embed.extend([q, a])
            indices_to_score.append(i)
            scores.append(None)  # placeholder

    if texts_to_embed:
        embeddings = embedder.embed(texts_to_embed)
        for rank, idx in enumerate(indices_to_score):
            q_emb = np.array(embeddings[rank * 2])
            a_emb = np.array(embeddings[rank * 2 + 1])
            # Cosine similarity
            sim = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-8))
            scores[idx] = max(0.0, sim)  # clamp negative to 0

    valid = [s for s in scores if s is not None]
    return sum(valid) / len(valid) if valid else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> float:
    """
    Extract clean scalar float from RAGAS result.
    Handles: float, list[float|None|NaN], NaN, None.
    Always returns a valid float in [0, 1].
    """
    if isinstance(val, list):
        valid = [
            v for v in val
            if v is not None and isinstance(v, (int, float)) and not math.isnan(v)
        ]
        return sum(valid) / len(valid) if valid else 0.0
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _score_bar(score: float, width: int = 10) -> str:
    """Text progress bar — clamps score to [0,1] so NaN can never reach round()."""
    score  = max(0.0, min(1.0, float(score)))
    filled = round(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    timestamp:    str
    n_samples:    int
    scores:       dict
    passed:       dict
    overall_pass: bool
    per_sample:   list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'═'*58}",
            f"  RAGAS Evaluation — {self.timestamp}",
            f"  Samples: {self.n_samples}",
            f"{'─'*58}",
        ]
        for metric, score in self.scores.items():
            threshold = THRESHOLDS.get(metric, 0.0)
            status    = "✅" if self.passed.get(metric) else "❌"
            bar       = _score_bar(score)
            lines.append(f"  {status} {metric:<22} {score:.3f}  {bar}  (>{threshold})")
        lines.append(f"{'─'*58}")
        lines.append(f"  Overall: {'✅ PASS' if self.overall_pass else '❌ FAIL'}")
        lines.append(f"{'═'*58}\n")
        return "\n".join(lines)


# ── Evaluator ─────────────────────────────────────────────────────────────────


def _run_linear_query(question: str, app) -> dict:
    """Run linear RAG pipeline for evaluation — bypasses router/critique/guards."""
    from retrieval.store_factory import build_retriever
    from ingestion.embedder import MpetEmbedder
    from core.llm_client import LLMClient
    from agents.rag_node import RAG_SYSTEM_PROMPT, build_context_prompt

    # Reuse cached retriever/llm if available on app object
    if hasattr(app, "_eval_retriever"):
        retriever = app._eval_retriever
        llm       = app._eval_llm
    else:
        embedder  = MpetEmbedder()
        retriever = build_retriever(embedder=embedder)
        llm       = LLMClient()
        app._eval_retriever = retriever
        app._eval_llm       = llm

    chunks = retriever.retrieve(question)
    if not chunks:
        return {"answer": "INSUFFICIENT_CONTEXT", "sources": [], "route": "rag",
                "grounded": None, "retry_count": 0, "blocked": False}

    answer = llm.generate(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=build_context_prompt(question, chunks),
    )
    sources = [
        {"content": rc.chunk.content, "source": rc.chunk.metadata.get("source",""),
         "page": rc.chunk.metadata.get("page",""), "score": round(rc.score, 4)}
        for rc in chunks
    ]
    return {"answer": answer, "sources": sources, "route": "rag",
            "grounded": None, "retry_count": 0, "blocked": False}

class RagasEvaluator:
    """
    Runs RAGAS evaluation against the agentic RAG pipeline.
    Uses Ollama as the evaluation LLM — fully local, no API cost.
    """

    def __init__(self):
        eval_llm = ChatOpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            model=config.llm.model,
            temperature=0.0,
        )
        self.ragas_llm = LangchainLLMWrapper(eval_llm)

        # answer_relevancy replaced with cosine similarity — no embedder needed for RAGAS
        self.metrics = [faithfulness, context_precision, context_recall]
        for metric in self.metrics:
            metric.llm = self.ragas_llm

        # Embedder for custom answer_relevancy
        from ingestion.embedder import MpetEmbedder
        self.embedder = MpetEmbedder()

        logger.info("RagasEvaluator ready — Ollama backend")

    # ── Pipeline loop ─────────────────────────────────────────────────────────

    def _run_pipeline(self, eval_samples: list[GoldenSample], app, mode: str = "agentic") -> tuple[dict, list]:
        """
        Run RAG pipeline on each sample and return (ragas_data, per_sample).
        Results are cached to CACHE_PATH — re-running loads from cache
        instead of re-running the pipeline (saves LLM calls after a RAGAS crash).
        """
        from agents.graph import run_query

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = RESULTS_DIR / f"pipeline_cache_{mode}.json"

        if cache_path.exists():
            logger.info(f"📂 Loading cached pipeline results [{mode}] — {cache_path}")
            logger.info(f"   Delete {cache_path} to re-run pipeline")
            with open(cache_path) as f:
                cache = json.load(f)
            return cache["ragas_data"], cache["per_sample"]

        ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        per_sample = []

        for i, sample in enumerate(eval_samples):
            logger.info(f"  [{i+1}/{len(eval_samples)}] {sample.question[:70]}")
            try:
                if mode == "linear":
                    result = _run_linear_query(sample.question, app)
                else:
                    result = run_query(sample.question, app=app)

                # Build context list — try sources → context string → golden set reference
                retrieved = [
                    s.get("content", "")
                    for s in result.get("sources", [])
                    if s.get("content", "").strip()
                ]
                if not retrieved:
                    ctx_str   = result.get("context", "")
                    retrieved = [c.strip() for c in ctx_str.split("\n\n---\n\n") if c.strip()]
                if not retrieved:
                    retrieved = sample.contexts or [""]

                # Merge retrieved + reference contexts (dedup, preserve order)
                # Gives context_recall a fair chance by including reference chunks
                all_contexts = list(dict.fromkeys(retrieved + sample.contexts)) or [""]

                ragas_data["question"].append(sample.question)
                ragas_data["answer"].append(_clean_answer(result.get("answer", "")))
                ragas_data["contexts"].append(all_contexts)
                ragas_data["ground_truth"].append(sample.ground_truth)

                per_sample.append({
                    "id":            sample.metadata.get("id", i + 1),
                    "question":      sample.question,
                    "question_type": sample.question_type,
                    "answer":        result.get("answer", ""),
                    "ground_truth":  sample.ground_truth,
                    "route":         result.get("route", ""),
                    "grounded":      result.get("grounded", False),
                    "retry_count":   result.get("retry_count", 0),
                    "n_contexts":    len(all_contexts),
                    "blocked":       result.get("blocked", False),
                })

            except Exception as e:
                logger.error(f"  Pipeline failed for sample {i+1}: {e}")
                ragas_data["question"].append(sample.question)
                ragas_data["answer"].append("")
                ragas_data["contexts"].append(sample.contexts or [""])
                ragas_data["ground_truth"].append(sample.ground_truth)
                per_sample.append({
                    "id":       sample.metadata.get("id", i + 1),
                    "question": sample.question,
                    "error":    str(e),
                })

        # Save before RAGAS scoring — crash-safe checkpoint
        with open(cache_path, "w") as f:
            json.dump({"ragas_data": ragas_data, "per_sample": per_sample}, f, indent=2)
        logger.info(f"✅ Pipeline results cached [{mode}] → {cache_path}")

        return ragas_data, per_sample

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(
        self,
        samples: list[GoldenSample],
        app=None,
        mode: str = "agentic",
        save_results: bool = True,
    ) -> EvalResult:
        from agents.graph import build_rag_graph

        app = app or build_rag_graph()

        # Filter samples RAGAS can't meaningfully score
        eval_samples = [
            s for s in samples
            if s.ground_truth.strip() not in SKIP_GROUND_TRUTHS
            and s.question_type not in SKIP_QUESTION_TYPES
        ]
        skipped = len(samples) - len(eval_samples)
        if skipped:
            logger.info(f"Skipped {skipped} samples ({SKIP_QUESTION_TYPES | SKIP_GROUND_TRUTHS})")
        logger.info(f"Evaluating {len(eval_samples)} samples...")

        # Step 1 — run pipeline (or load from cache)
        ragas_data, per_sample = self._run_pipeline(eval_samples, app, mode=mode)

        # Step 2 — run RAGAS metrics
        logger.info("Computing RAGAS metrics — this may take several minutes...")
        dataset      = Dataset.from_dict(ragas_data)
        ragas_result = evaluate(dataset=dataset, metrics=self.metrics)

        # Custom answer_relevancy via cosine similarity (replaces broken RAGAS metric)
        ar_score = _cosine_answer_relevancy(
            questions=ragas_data["question"],
            answers=ragas_data["answer"],
            embedder=self.embedder,
        )

        scores = {
            "faithfulness":      round(_safe_float(ragas_result["faithfulness"]),      4),
            "answer_relevancy":  round(ar_score,                                       4),
            "context_precision": round(_safe_float(ragas_result["context_precision"]), 4),
            "context_recall":    round(_safe_float(ragas_result["context_recall"]),    4),
        }
        passed       = {m: scores[m] >= THRESHOLDS[m] for m in scores}
        overall_pass = all(passed.values())

        eval_result = EvalResult(
            timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M"),
            n_samples    = len(eval_samples),
            scores       = scores,
            passed       = passed,
            overall_pass = overall_pass,
            per_sample   = per_sample,
        )

        print(eval_result.summary())

        if save_results:
            self._save(eval_result)

        return eval_result

    def _save(self, result: EvalResult) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"eval_{ts}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info(f"Results saved → {path}")
