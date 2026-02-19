"""
Input Guards — validate the user query BEFORE it enters the pipeline.

Guards run in order, first BLOCK wins.
Each guard has a single responsibility — easy to add/remove/reorder.

Included guards:
  PromptInjectionGuard  — detects jailbreak / instruction override attempts
  TopicRelevanceGuard   — blocks off-topic queries (configurable)
  InputLengthGuard      — rejects queries that are too short or too long
  PiiDetectionGuard     — detects PII in the input query (warns, doesn't block)
"""
import logging
import re

from guardrails.base import GuardBase, GuardResult, GuardAction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Injection Guard
# ---------------------------------------------------------------------------

# Common jailbreak / prompt injection patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(your\s+)?(previous|prior|system)\s+(prompt|instructions?)",
    r"you\s+are\s+now\s+(a\s+)?(\w+\s+)?without\s+(any\s+)?restrictions?",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"act\s+as\s+(if\s+you\s+are\s+)?(?:dan|jailbreak|unrestricted|evil)",
    r"forget\s+(everything|all)\s+(you\s+)?(know|were\s+told)",
    r"override\s+(your\s+)?(safety|content)\s+(filter|policy|guidelines?)",
    r"system\s*:\s*you\s+are",          # Fake system prompt injection
    r"<\s*system\s*>",                  # XML-style system tag injection
    r"\[\s*system\s*\]",                # Bracket-style injection
    r"new\s+instructions?\s*:",
    r"your\s+real\s+instructions?\s+(are|say)",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


class PromptInjectionGuard(GuardBase):
    name = "prompt_injection"

    def check(self, text: str, context: dict | None = None) -> GuardResult:
        for pattern in _INJECTION_RE:
            if pattern.search(text):
                logger.warning(f"[PromptInjectionGuard] Injection detected: '{text[:80]}'")
                return GuardResult(
                    action=GuardAction.BLOCK,
                    reason="Prompt injection attempt detected. Please ask a genuine question.",
                )
        return GuardResult(action=GuardAction.PASS, reason="No injection detected")


# ---------------------------------------------------------------------------
# Topic Relevance Guard
# ---------------------------------------------------------------------------

class TopicRelevanceGuard(GuardBase):
    """
    Blocks queries clearly outside your domain.
    Configure allowed_topics and blocked_topics for your use case.
    Default: permissive (warns only) — tighten for enterprise deployments.
    """
    name = "topic_relevance"

    def __init__(
        self,
        blocked_topics: list[str] | None = None,
        warn_only: bool = True,
    ):
        # Default blocked topics — override for your domain
        self.blocked_topics = blocked_topics or [
            "how to make weapons",
            "how to hack",
            "illegal activities",
            "self harm",
            "explicit content",
        ]
        self.warn_only = warn_only  # True = warn and continue, False = block
        self._patterns = [
            re.compile(re.escape(t), re.IGNORECASE)
            for t in self.blocked_topics
        ]

    def check(self, text: str, context: dict | None = None) -> GuardResult:
        for pattern in self._patterns:
            if pattern.search(text):
                action = GuardAction.WARN if self.warn_only else GuardAction.BLOCK
                reason = f"Query touches a restricted topic. Flagged for review."
                logger.warning(f"[TopicRelevanceGuard] Flagged: '{text[:80]}'")
                return GuardResult(action=action, reason=reason)
        return GuardResult(action=GuardAction.PASS, reason="Topic is acceptable")


# ---------------------------------------------------------------------------
# Input Length Guard
# ---------------------------------------------------------------------------

class InputLengthGuard(GuardBase):
    name = "input_length"

    def __init__(self, min_chars: int = 3, max_chars: int = 2000):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def check(self, text: str, context: dict | None = None) -> GuardResult:
        length = len(text.strip())
        if length < self.min_chars:
            return GuardResult(
                action=GuardAction.BLOCK,
                reason=f"Query too short ({length} chars). Please provide more detail.",
            )
        if length > self.max_chars:
            return GuardResult(
                action=GuardAction.BLOCK,
                reason=f"Query too long ({length} chars, max {self.max_chars}). Please be more concise.",
            )
        return GuardResult(action=GuardAction.PASS, reason="Length is acceptable")


# ---------------------------------------------------------------------------
# PII Detection Guard (input)
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone":       r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "ip_address":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

_PII_RE = {name: re.compile(pattern) for name, pattern in _PII_PATTERNS.items()}


class PiiDetectionGuard(GuardBase):
    """
    Detects PII in the input query.
    Default: WARN only (log but continue) — set block_on_pii=True to hard block.
    """
    name = "pii_detection_input"

    def __init__(self, block_on_pii: bool = False):
        self.block_on_pii = block_on_pii

    def check(self, text: str, context: dict | None = None) -> GuardResult:
        found = [name for name, pattern in _PII_RE.items() if pattern.search(text)]
        if found:
            action = GuardAction.BLOCK if self.block_on_pii else GuardAction.WARN
            reason = f"PII detected in query: {', '.join(found)}"
            logger.warning(f"[PiiDetectionGuard] {reason}")
            return GuardResult(action=action, reason=reason, metadata={"pii_types": found})
        return GuardResult(action=GuardAction.PASS, reason="No PII detected")


# ---------------------------------------------------------------------------
# Default input guard stack
# ---------------------------------------------------------------------------

DEFAULT_INPUT_GUARDS: list[GuardBase] = [
    InputLengthGuard(),
    PromptInjectionGuard(),
    TopicRelevanceGuard(warn_only=True),
    PiiDetectionGuard(block_on_pii=False),
]
