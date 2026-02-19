"""
Guardrails interfaces and shared data models.
Keeps guardrails plug-n-play — swap rule-based for LLM-based validators
by implementing GuardBase.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class GuardAction(Enum):
    PASS    = "pass"     # Continue pipeline normally
    BLOCK   = "block"    # Hard stop — return safe response to user
    REDACT  = "redact"   # Modify content and continue
    WARN    = "warn"     # Log warning but continue


@dataclass
class GuardResult:
    action:   GuardAction
    reason:   str                        # Human-readable explanation
    modified: str | None = None          # Redacted/modified content (if action=REDACT)
    metadata: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.action in (GuardAction.PASS, GuardAction.WARN, GuardAction.REDACT)

    @property
    def blocked(self) -> bool:
        return self.action == GuardAction.BLOCK


class GuardBase(ABC):
    """
    Implement this to add a custom input or output validator.
    One guard = one responsibility (single responsibility principle).
    """
    name: str = "unnamed_guard"

    @abstractmethod
    def check(self, text: str, context: dict | None = None) -> GuardResult:
        """
        Validate text. Return GuardResult indicating what to do next.
        context: optional dict with extra info (query, sources, etc.)
        """
        ...
