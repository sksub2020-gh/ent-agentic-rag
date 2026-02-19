"""
Agent state — the single shared object flowing through all LangGraph nodes.
Every node reads from and writes to this TypedDict.
Keeping state explicit makes debugging and LangSmith tracing trivial.
"""
from typing import Literal
from typing_extensions import TypedDict

from core.interfaces import RetrievedChunk


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str                          # Original user query

    # ── Router node output ─────────────────────────────────────────────────
    route: Literal["rag", "direct"]     # "rag" = retrieve first, "direct" = LLM knows it
    router_reasoning: str               # Why router made this decision

    # ── RAG node output ────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]   # Reranked chunks from HybridRetriever
    context: str                             # Formatted context string for LLM

    # ── Critique node output ───────────────────────────────────────────────
    answer: str                         # Final generated answer
    grounded: bool                      # Is answer supported by retrieved context?
    critique_reasoning: str             # Why critique passed/failed
    retry_count: int                    # How many times we've retried retrieval

    # ── Metadata ───────────────────────────────────────────────────────────
    sources: list[dict]                 # [{source, page, score}] for citation display

    # ── Guardrails (Phase 6) ───────────────────────────────────────────────
    blocked: bool                       # True if any guard hard-blocked this request
    block_reason: str | None            # Which guard blocked and why
    guard_warnings: list[str]           # Non-blocking warnings from guards
