"""
Golden set loader — loads hand-curated Q&A pairs from golden_set.json.
Schema matches the uploaded golden_set.json format.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = "./evaluation/golden_set.json"


@dataclass
class GoldenSample:
    """One Q&A pair from the golden set."""
    question:              str
    ground_truth:          str                # expected answer
    contexts:              list[str]          # reference chunk texts
    question_type:         str  = "factual"
    metadata:              dict = field(default_factory=dict)


def load_golden_set(
    path: str = GOLDEN_SET_PATH,
    exclude_types: list[str] | None = None,
) -> list[GoldenSample]:
    """
    Load golden set from JSON.

    Args:
        path:          Path to golden_set.json
        exclude_types: Question types to skip (e.g. ["Irrelevant"] to skip
                       general knowledge questions unrelated to the corpus)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden set not found: {p}. Create evaluation/golden_set.json first.")

    with open(p) as f:
        data = json.load(f)

    samples = []
    for item in data:
        qtype = item.get("metadata", {}).get("question_type", "factual")
        if exclude_types and qtype in exclude_types:
            continue
        samples.append(GoldenSample(
            question      = item["question"],
            ground_truth  = item["ground_truth"],
            contexts      = item.get("contexts", []),
            question_type = qtype,
            metadata      = item.get("metadata", {}),
        ))

    logger.info(f"Loaded {len(samples)} golden samples from {p}")
    return samples


def summarise(samples: list[GoldenSample]) -> None:
    """Print a summary of the golden set composition."""
    from collections import Counter
    types = Counter(s.question_type for s in samples)
    print(f"\nGolden set: {len(samples)} samples")
    for qtype, count in sorted(types.items()):
        print(f"  {qtype:<20} {count}")
    print()
