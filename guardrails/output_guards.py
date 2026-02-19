"""
Output Guards — validate the LLM answer BEFORE it reaches the user.

Included guards:
  PiiRedactionGuard     — redacts PII from the answer (REDACT action)
  ToxicityGuard         — blocks toxic / harmful answers
  HallucinationFlagGuard — flags answers containing phrases that signal fabrication
  AnswerCompletenessGuard — blocks empty or suspiciously short answers
"""
import logging
import re

from guardrails.base import GuardBase, GuardResult, GuardAction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII Redaction Guard
# ---------------------------------------------------------------------------

_PII_REDACTION_PATTERNS = {
    "email":       (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL REDACTED]"),
    "phone":       (r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE REDACTED]"),
    "ssn":         (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]"),
    "credit_card": (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD REDACTED]"),
}


class PiiRedactionGuard(GuardBase):
    """
    Redacts PII from LLM output.
    Uses REDACT action — content is modified but pipeline continues.
    """
    name = "pii_redaction_output"

    def check(self, text: str, context: dict | None = None) -> GuardResult:
        redacted = text
        found = []
        for pii_type, (pattern, replacement) in _PII_REDACTION_PATTERNS.items():
            new_text = re.sub(pattern, replacement, redacted)
            if new_text != redacted:
                found.append(pii_type)
                redacted = new_text

        if found:
            logger.warning(f"[PiiRedactionGuard] Redacted PII from output: {found}")
            return GuardResult(
                action=GuardAction.REDACT,
                reason=f"PII redacted from answer: {', '.join(found)}",
                modified=redacted,
                metadata={"pii_types": found},
            )
        return GuardResult(action=GuardAction.PASS, reason="No PII in output")


# ---------------------------------------------------------------------------
# Toxicity Guard
# ---------------------------------------------------------------------------

# Lightweight keyword-based toxicity check.
# For production: replace with a dedicated model (e.g. Detoxify, Perspective API).
_TOXIC_PATTERNS = [
    r"\b(kill|murder|assault|rape|torture)\s+(yourself|themselves|people)\b",
    r"\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive|poison)\b",
    r"\bstep[s\s]+to\s+(commit|perform)\s+(suicide|self.harm)\b",
]
_TOXIC_RE = [re.compile(p, re.IGNORECASE) for p in _TOXIC_PATTERNS]


class ToxicityGuard(GuardBase):
    name = "toxicity"

    def check(self, text: str, context: dict | None = None) -> GuardResult:
        for pattern in _TOXIC_RE:
            if pattern.search(text):
                logger.error(f"[ToxicityGuard] Toxic content detected in output")
                return GuardResult(
                    action=GuardAction.BLOCK,
                    reason="Answer contains harmful content and cannot be shown.",
                )
        return GuardResult(action=GuardAction.PASS, reason="No toxic content detected")


# ---------------------------------------------------------------------------
# Hallucination Flag Guard
# ---------------------------------------------------------------------------

# Phrases that often signal the LLM is making things up
_HALLUCINATION_PHRASES = [
    r"as (of|per) my (knowledge|training|understanding)",
    r"i (believe|think|assume) (that )?",
    r"it (is|seems) (likely|possible|probable) that",
    r"based on (my|general) knowledge",
    r"i('m| am) not (entirely |completely )?sure (but|,)",
    r"(generally|typically|usually) speaking",
    r"in most cases",
]
_HALLUCINATION_RE = [re.compile(p, re.IGNORECASE) for p in _HALLUCINATION_PHRASES]


class HallucinationFlagGuard(GuardBase):
    """
    Flags answers that contain LLM uncertainty phrases — signals the model
    may be going beyond the retrieved context.
    Default: WARN (log but show answer). Set block=True to be strict.
    """
    name = "hallucination_flag"

    def __init__(self, block: bool = False):
        self.block = block

    def check(self, text: str, context: dict | None = None) -> GuardResult:
        triggered = [p.pattern for p in _HALLUCINATION_RE if p.search(text)]
        if triggered:
            action = GuardAction.BLOCK if self.block else GuardAction.WARN
            reason = "Answer contains uncertainty phrases that may indicate hallucination."
            logger.warning(f"[HallucinationFlagGuard] {reason} Patterns: {triggered[:2]}")
            return GuardResult(action=action, reason=reason, metadata={"patterns": triggered})
        return GuardResult(action=GuardAction.PASS, reason="No hallucination signals detected")


# ---------------------------------------------------------------------------
# Answer Completeness Guard
# ---------------------------------------------------------------------------

class AnswerCompletenessGuard(GuardBase):
    """Blocks empty or suspiciously short answers."""
    name = "answer_completeness"

    def __init__(self, min_chars: int = 20):
        self.min_chars = min_chars

    def check(self, text: str, context: dict | None = None) -> GuardResult:
        stripped = text.strip()
        if not stripped:
            return GuardResult(
                action=GuardAction.BLOCK,
                reason="Answer is empty — pipeline failed to generate a response.",
            )
        if len(stripped) < self.min_chars:
            return GuardResult(
                action=GuardAction.WARN,
                reason=f"Answer is very short ({len(stripped)} chars) — may be incomplete.",
            )
        return GuardResult(action=GuardAction.PASS, reason="Answer length is acceptable")


# ---------------------------------------------------------------------------
# Default output guard stack
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_GUARDS: list[GuardBase] = [
    AnswerCompletenessGuard(),
    ToxicityGuard(),
    PiiRedactionGuard(),
    HallucinationFlagGuard(block=False),   # Set block=True for stricter deployments
]
