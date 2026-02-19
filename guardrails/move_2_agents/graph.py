"""
LangGraph Graph Builder — wires guardrails + Router → RAG → Critique pipeline.
LangSmith tracing is enabled automatically via .env variables (no code changes needed).

Graph structure:
  START
    ↓
  [input_guard]   — validates query (injection, PII, topic, length)
    ↓ (conditional)
    ├─ blocked → END  (short-circuit, skip pipeline entirely)
    └─ pass    ↓
  [router_node]   — decides "rag" or "direct"
    ↓
  [rag_node]      — retrieves + generates
    ↓
  [critique_node] — checks grounding
    ↓ (conditional)
    ├─ grounded=False → back to [rag_node] (max 2 retries)
    └─ grounded=True  ↓
  [output_guard]  — redacts PII, blocks toxicity, flags hallucination
    ↓
  END
"""

import logging
from functools import partial

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.router_node import router_node
from agents.rag_node import rag_node
from agents.critique_node import critique_node, should_retry
from guardrails.guard_runner import (
    GuardRunner,
    input_guard_node,
    output_guard_node,
    should_continue_after_input_guard,
)
from guardrails.input_guards import DEFAULT_INPUT_GUARDS
from guardrails.output_guards import DEFAULT_OUTPUT_GUARDS
from core.llm_client import LLMClient
from core.interfaces import LLMClientBase
from retrieval.hybrid_retriever import HybridRetriever
from ingestion.embedder import MpetEmbedder
from retrieval.milvus_store import MilvusLiteStore
from retrieval.bm25_store import BM25SStore
from retrieval.hybrid_retriever import FlashRankReranker

logger = logging.getLogger(__name__)


def build_rag_graph(
    llm: LLMClientBase | None = None,
    retriever: HybridRetriever | None = None,
    input_guards: list | None = None,
    output_guards: list | None = None,
) -> StateGraph:
    """
    Builds and compiles the Agentic RAG LangGraph with guardrails.

    Args:
        llm:           LLMClientBase instance (defaults to OllamaClient)
        retriever:     HybridRetriever instance (defaults to full local stack)
        input_guards:  List of GuardBase for input validation (defaults to DEFAULT_INPUT_GUARDS)
        output_guards: List of GuardBase for output validation (defaults to DEFAULT_OUTPUT_GUARDS)

    Returns:
        Compiled LangGraph app ready for .invoke() / .stream()
    """
    # ── Instantiate dependencies if not injected ──────────────────────────
    llm = llm or LLMClient()
    retriever = retriever or HybridRetriever(
        vector_store=MilvusLiteStore(),
        sparse_store=BM25SStore(),
        embedder=MpetEmbedder(),
        reranker=FlashRankReranker(),
    )
    input_runner = GuardRunner(input_guards or DEFAULT_INPUT_GUARDS)
    output_runner = GuardRunner(output_guards or DEFAULT_OUTPUT_GUARDS)

    # ── Bind dependencies to nodes via partial ────────────────────────────
    bound_input_guard = partial(input_guard_node, runner=input_runner)
    bound_router = partial(router_node, llm=llm)
    bound_rag = partial(rag_node, llm=llm, retriever=retriever)
    bound_critique = partial(critique_node, llm=llm)
    bound_output_guard = partial(output_guard_node, runner=output_runner)

    # ── Build graph ───────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("input_guard", bound_input_guard)
    graph.add_node("router", bound_router)
    graph.add_node("rag", bound_rag)
    graph.add_node("critique", bound_critique)
    graph.add_node("output_guard", bound_output_guard)

    # Entry point
    graph.set_entry_point("input_guard")

    # Input guard → conditional: short-circuit if blocked, else continue to router
    graph.add_conditional_edges(
        "input_guard",
        should_continue_after_input_guard,
        {
            "end": END,  # Blocked — skip everything
            "continue": "router",  # Pass — proceed normally
        },
    )

    # Router → RAG
    graph.add_edge("router", "rag")

    # RAG → Critique
    graph.add_edge("rag", "critique")

    # Critique → conditional: retry or output guard
    graph.add_conditional_edges(
        "critique",
        should_retry,
        {
            "retry": "rag",  # Re-retrieve and regenerate
            "end": "output_guard",  # Done with RAG — validate output
        },
    )

    # Output guard → END
    graph.add_edge("output_guard", END)

    logger.info(
        "LangGraph compiled ✓ — "
        "input_guard → router → rag → critique → output_guard → END"
    )
    return graph.compile()


def run_query(query: str, app=None) -> dict:
    """
    Run a single query through the agentic RAG graph.

    Args:
        query: User's question
        app:   Pre-built graph (built once, reused for multiple queries)

    Returns:
        Final AgentState dict with answer, sources, grounding info
    """
    if app is None:
        app = build_rag_graph()

    initial_state: AgentState = {
        "query": query,
        "route": "rag",
        "router_reasoning": "",
        "retrieved_chunks": [],
        "context": "",
        "answer": "",
        "grounded": False,
        "critique_reasoning": "",
        "retry_count": 0,
        "sources": [],
        "blocked": False,
        "block_reason": None,
        "guard_warnings": [],
    }

    logger.info(f"[Graph] Running query: '{query[:80]}'")
    final_state = app.invoke(initial_state)

    return {
        "query": final_state["query"],
        "answer": final_state["answer"],
        "route": final_state.get("route", ""),
        "router_reasoning": final_state.get("router_reasoning", ""),
        "grounded": final_state.get("grounded", False),
        "critique_reasoning": final_state.get("critique_reasoning", ""),
        "retry_count": final_state.get("retry_count", 0),
        "sources": final_state.get("sources", []),
        "blocked": final_state.get("blocked", False),
        "block_reason": final_state.get("block_reason"),
        "guard_warnings": final_state.get("guard_warnings", []),
    }
