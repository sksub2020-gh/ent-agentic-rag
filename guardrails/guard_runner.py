"""
Guard Runner + LangGraph nodes for input and output guardrails.

GuardRunner: runs a list of guards in order, first BLOCK wins.
Two LangGraph nodes:
  input_guard_node  — sits before router, validates query
  output_guard_node — sits after critique, validates answer
"""
import logging
from dataclasses import dataclass, field

from guardrails.base import GuardBase, GuardResult, GuardAction

logger = logging.getLogger(__name__)

# Safe fallback responses shown to users when a guard blocks
_BLOCK_RESPONSES = {
    "input":  "I'm unable to process this request. Please rephrase your question.",
    "output": "I was unable to generate a safe response. Please try rephrasing your question.",
}


@dataclass
class RunnerResult:
    """Aggregated result from running all guards in a stack."""
    passed: bool
    blocked_by: str | None = None       # Name of the guard that blocked
    block_reason: str | None = None
    redacted_text: str | None = None    # Final text after any REDACT actions
    warnings: list[str] = field(default_factory=list)
    all_results: list[GuardResult] = field(default_factory=list)


class GuardRunner:
    """
    Runs guards in sequence. First BLOCK stops execution.
    REDACT actions accumulate — text is progressively cleaned.
    WARN actions are collected but don't stop the pipeline.
    """

    def __init__(self, guards: list[GuardBase]):
        self.guards = guards

    def run(self, text: str, context: dict | None = None) -> RunnerResult:
        current_text = text
        warnings = []
        all_results = []

        for guard in self.guards:
            result = guard.check(current_text, context=context)
            all_results.append(result)

            if result.action == GuardAction.BLOCK:
                logger.warning(f"[GuardRunner] BLOCKED by '{guard.name}': {result.reason}")
                return RunnerResult(
                    passed=False,
                    blocked_by=guard.name,
                    block_reason=result.reason,
                    warnings=warnings,
                    all_results=all_results,
                )

            elif result.action == GuardAction.REDACT:
                logger.info(f"[GuardRunner] REDACTED by '{guard.name}': {result.reason}")
                current_text = result.modified or current_text

            elif result.action == GuardAction.WARN:
                logger.warning(f"[GuardRunner] WARN from '{guard.name}': {result.reason}")
                warnings.append(f"{guard.name}: {result.reason}")

        return RunnerResult(
            passed=True,
            redacted_text=current_text if current_text != text else None,
            warnings=warnings,
            all_results=all_results,
        )


# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------

def input_guard_node(state: dict, runner: GuardRunner) -> dict:
    """
    LangGraph node: Input Guard.
    Reads:  state["query"]
    Writes: state["query"] (may be unchanged),
            state["blocked"] (True if guard blocked),
            state["block_reason"],
            state["guard_warnings"]

    Sits BEFORE router_node in the graph.
    """
    query = state["query"]
    logger.info(f"[InputGuard] Checking query: '{query[:80]}'")

    result = runner.run(query, context={"stage": "input"})

    if not result.passed:
        return {
            **state,
            "answer":       _BLOCK_RESPONSES["input"],
            "blocked":      True,
            "block_reason": result.block_reason,
            "guard_warnings": result.warnings,
        }

    return {
        **state,
        "blocked":        False,
        "block_reason":   None,
        "guard_warnings": result.warnings,
    }


def output_guard_node(state: dict, runner: GuardRunner) -> dict:
    """
    LangGraph node: Output Guard.
    Reads:  state["answer"], state["query"], state["sources"]
    Writes: state["answer"] (may be redacted),
            state["blocked"],
            state["block_reason"],
            state["guard_warnings"]

    Sits AFTER critique_node in the graph.
    """
    answer = state.get("answer", "")
    logger.info(f"[OutputGuard] Checking answer ({len(answer)} chars)")

    result = runner.run(
        answer,
        context={"query": state.get("query"), "sources": state.get("sources")},
    )

    if not result.passed:
        return {
            **state,
            "answer":       _BLOCK_RESPONSES["output"],
            "blocked":      True,
            "block_reason": result.block_reason,
            "guard_warnings": state.get("guard_warnings", []) + result.warnings,
        }

    # Apply any redactions
    final_answer = result.redacted_text or answer

    return {
        **state,
        "answer":         final_answer,
        "blocked":        False,
        "block_reason":   None,
        "guard_warnings": state.get("guard_warnings", []) + result.warnings,
    }


def should_continue_after_input_guard(state: dict) -> str:
    """
    Conditional edge after input_guard_node.
    Blocked queries short-circuit directly to END — skip router/rag/critique.
    """
    return "end" if state.get("blocked") else "continue"
