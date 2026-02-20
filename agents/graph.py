"""
LangGraph Graph Builder — wires Router → RAG → Critique with conditional retry loop.
LangSmith tracing is enabled automatically via .env variables (no code changes needed).

Graph structure:
  START
    ↓
  [router_node]  — decides "rag" or "direct"
    ↓
  [rag_node]     — retrieves + generates
    ↓
  [critique_node] — checks grounding
    ↓ (conditional)
    ├─ grounded=True  → END
    └─ grounded=False → back to [rag_node] (max 2 retries)
"""

import logging
from functools import partial

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.router_node import router_node
from agents.rag_node import rag_node
from agents.critique_node import critique_node, should_retry
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
) -> StateGraph:
    """
    Builds and compiles the Agentic RAG LangGraph.

    Args:
        llm: LLMClientBase instance (defaults to OllamaClient)
        retriever: HybridRetriever instance (defaults to full local stack)

    Returns:
        Compiled LangGraph app ready for .invoke() / .stream()
    """
    # ── Instantiate dependencies if not injected ──────────────────────────
    llm = llm or LLMClient()
    embedder=MpetEmbedder()
    retriever = retriever or HybridRetriever(
        embedder=embedder,
        vector_store=MilvusLiteStore(embedder.dimension),
        sparse_store=BM25SStore(),
        reranker=FlashRankReranker(),
    )

    # ── Bind dependencies to nodes via partial (keeps nodes pure functions) ─
    # Why partial? Nodes stay testable in isolation — just call node(state, llm=mock_llm)
    bound_router = partial(router_node, llm=llm)
    bound_rag = partial(rag_node, llm=llm, retriever=retriever)
    bound_critique = partial(critique_node, llm=llm)

    # ── Build graph ──────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", bound_router)
    graph.add_node("rag", bound_rag)
    graph.add_node("critique", bound_critique)

    # Entry point
    graph.set_entry_point("router")

    # Router → RAG (always, regardless of route — rag_node handles "direct" internally)
    graph.add_edge("router", "rag")

    # RAG → Critique
    graph.add_edge("rag", "critique")

    # Critique → conditional: retry or end
    graph.add_conditional_edges(
        "critique",
        should_retry,  # Returns "retry" or "end"
        {
            "retry": "rag",  # Back to RAG node for another attempt
            "end": END,  # Done
        },
    )

    logger.info("LangGraph compiled ✓ — nodes: router → rag → critique → (retry | end)")
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
    }

    logger.info(f"[Graph] Running query: '{query[:80]}'")
    final_state = app.invoke(initial_state)

    return {
        "query": final_state["query"],
        "answer": final_state["answer"],
        "route": final_state["route"],
        "router_reasoning": final_state["router_reasoning"],
        "grounded": final_state["grounded"],
        "critique_reasoning": final_state["critique_reasoning"],
        "retry_count": final_state["retry_count"],
        "sources": final_state["sources"],
    }
