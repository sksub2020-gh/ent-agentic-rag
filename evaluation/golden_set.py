"""
Golden Test Set Builder — generates Q&A pairs from your ingested documents.

Two modes:
  1. Auto-generate: uses LLM to create Q&A pairs from chunks (fast bootstrap)
  2. Manual: loads from a hand-curated CSV/JSON (gold standard)

A good golden set has 20-30 pairs covering:
  - Simple factual lookups
  - Multi-hop questions (answer spans multiple chunks)
  - Edge cases (questions with no answer in corpus)
"""
import csv
import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

from core.interfaces import LLMClientBase, RetrievedChunk
from retrieval.milvus_store import MilvusLiteStore
from retrieval.bm25_store import BM25SStore
from ingestion.embedder import MpetEmbedder

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = "./evaluation/golden_set.json"

QUESTION_GEN_SYSTEM_PROMPT = """You are an expert at creating evaluation datasets for RAG systems.
Given a text passage, generate realistic questions a user might ask whose answers are FULLY contained in the passage.

Respond with ONLY valid JSON — a list of objects:
[
  {"question": "...", "answer": "...", "question_type": "factual|inferential|edge_case"},
  ...
]

Rules:
- Generate 2-3 questions per passage
- Make questions specific, not vague
- Include at least one inferential question (requires reasoning, not just extraction)
- Answers must be fully supported by the passage — no outside knowledge
- question_type: "factual" (direct lookup), "inferential" (requires reasoning), "edge_case" (boundary/negative)"""


@dataclass
class GoldenSample:
    """One Q&A pair in the golden test set."""
    question: str
    ground_truth_answer: str
    ground_truth_contexts: list[str]    # The chunk contents that contain the answer
    source_docs: list[str]              # File paths of source documents
    question_type: str = "factual"      # factual | inferential | edge_case
    metadata: dict = field(default_factory=dict)


class GoldenSetBuilder:
    """
    Builds a golden test set for RAGAS evaluation.
    """

    def __init__(self, llm: LLMClientBase):
        self.llm = llm

    def generate_from_chunks(
        self,
        n_samples: int = 25,
        question_types: list[str] | None = None,
        seed: int = 42,
    ) -> list[GoldenSample]:
        """
        Auto-generate Q&A pairs by sampling chunks from the vector store
        and prompting the LLM to create questions from them.

        Args:
            n_samples:      Target number of Q&A pairs
            question_types: Filter to specific types (default: all)
            seed:           Random seed for reproducibility
        """
        random.seed(seed)

        # Pull chunks from BM25 store (has all corpus chunks in memory)
        sparse_store = BM25SStore()
        all_chunks = sparse_store._corpus_chunks

        if not all_chunks:
            raise ValueError(
                "No chunks found in BM25S store. Run ingestion_pipeline.py first."
            )

        logger.info(f"Generating golden set from {len(all_chunks)} chunks → target {n_samples} samples")

        # Sample chunks, prefer diversity across source docs
        sampled = self._diverse_sample(all_chunks, n=min(n_samples * 2, len(all_chunks)))

        samples: list[GoldenSample] = []
        for chunk in sampled:
            if len(samples) >= n_samples:
                break

            qa_pairs = self._generate_qa_from_chunk(chunk.content)
            for qa in qa_pairs:
                if len(samples) >= n_samples:
                    break
                qtype = qa.get("question_type", "factual")
                if question_types and qtype not in question_types:
                    continue

                samples.append(GoldenSample(
                    question=qa["question"],
                    ground_truth_answer=qa["answer"],
                    ground_truth_contexts=[chunk.content],
                    source_docs=[chunk.metadata.get("source", "")],
                    question_type=qtype,
                    metadata={"chunk_id": chunk.chunk_id},
                ))

        # Always add a few "no answer" edge cases
        samples.extend(self._make_no_answer_samples())

        logger.info(f"Generated {len(samples)} golden samples")
        return samples

    def load_from_file(self, path: str = GOLDEN_SET_PATH) -> list[GoldenSample]:
        """Load a hand-curated or previously saved golden set."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Golden set not found: {path}")

        with open(path) as f:
            data = json.load(f)

        samples = [GoldenSample(**d) for d in data]
        logger.info(f"Loaded {len(samples)} golden samples from {path}")
        return samples

    def save(self, samples: list[GoldenSample], path: str = GOLDEN_SET_PATH) -> None:
        """Persist golden set to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(s) for s in samples], f, indent=2)
        logger.info(f"Saved {len(samples)} golden samples → {path}")

    def export_csv(self, samples: list[GoldenSample], path: str = "./evaluation/golden_set.csv") -> None:
        """Export to CSV for easy human review and editing in Excel/Sheets."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "ground_truth_answer", "question_type",
                "source_docs", "ground_truth_contexts"
            ])
            writer.writeheader()
            for s in samples:
                writer.writerow({
                    "question": s.question,
                    "ground_truth_answer": s.ground_truth_answer,
                    "question_type": s.question_type,
                    "source_docs": "; ".join(s.source_docs),
                    "ground_truth_contexts": " | ".join(s.ground_truth_contexts),
                })
        logger.info(f"Exported CSV → {path} (review and edit before using as ground truth)")

    # ── Private helpers ────────────────────────────────────────────────────

    def _generate_qa_from_chunk(self, chunk_text: str) -> list[dict]:
        """Ask LLM to generate Q&A pairs from a chunk."""
        try:
            raw = self.llm.generate(
                system_prompt=QUESTION_GEN_SYSTEM_PROMPT,
                user_prompt=f"Passage:\n{chunk_text[:1500]}\n\nGenerate questions:",
            )
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except Exception as e:
            logger.warning(f"Q&A generation failed for chunk: {e}")
            return []

    def _diverse_sample(self, chunks, n: int):
        """Sample chunks across different source documents for diversity."""
        from collections import defaultdict
        by_source = defaultdict(list)
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            by_source[src].append(chunk)

        result = []
        sources = list(by_source.keys())
        while len(result) < n and any(by_source[s] for s in sources):
            for src in sources:
                if by_source[src] and len(result) < n:
                    pool = by_source[src]
                    result.append(pool.pop(random.randint(0, len(pool) - 1)))
        return result

    def _make_no_answer_samples(self) -> list[GoldenSample]:
        """Add negative samples — questions with no answer in the corpus."""
        return [
            GoldenSample(
                question="What is the population of Mars?",
                ground_truth_answer="I don't have enough information to answer this.",
                ground_truth_contexts=[],
                source_docs=[],
                question_type="edge_case",
                metadata={"note": "out-of-corpus question"},
            ),
        ]
