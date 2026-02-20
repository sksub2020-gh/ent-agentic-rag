"""
RAG Node — retrieves relevant chunks and generates a grounded answer.

Two responsibilities (kept in one node for efficiency):
  1. Retrieve: HybridRetriever → top-k reranked chunks
  2. Generate: format context + call LLM

The Critique node then validates the output.
"""
import logging

from core.interfaces import LLMClientBase, RetrievedChunk
from retrieval.hybrid_retriever import HybridRetriever
from agents.state import AgentState
from config.settings import config

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a precise, grounded assistant. Answer questions ONLY using the provided context chunks below.

Rules:
- READ ALL context chunks carefully before answering — the answer may be in any chunk.
- If ANY chunk contains the answer, use it. Do not ignore chunks just because they are short.
- If the context contains the answer, state it directly and cite the source.
- If the context is truly insufficient, say exactly: "INSUFFICIENT_CONTEXT"
- Never fabricate facts. Never use knowledge outside the provided context.
- Format citations as [Source: <filename>, Page: <page>] at the end of each claim.
- Be concise. Prefer bullet points for multi-part answers."""

DIRECT_SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant.
Answer the user's question directly and concisely from your general knowledge.
If you're unsure, say so clearly."""


def _format_context(chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    """
    Formats retrieved chunks into LLM-ready context string.
    Also returns structured sources list for citation display.
    Used internally by rag_node for the agentic pipeline.
    """
    context_blocks = []
    sources = []

    for i, rc in enumerate(chunks, 1):
        meta = rc.chunk.metadata
        source  = meta.get("source", "unknown")
        page    = meta.get("page", "?")
        section = meta.get("section", "")

        header = f"[{i}] {source} | Page {page}"
        if section:
            header += f" | {section}"

        context_blocks.append(f"{header}\n{rc.chunk.content}")
        sources.append({
            "index":   i,
            "source":  source,
            "page":    page,
            "section": section,
            "score":   round(rc.score, 4),
        })

    return "\n\n---\n\n".join(context_blocks), sources


def build_context_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into an LLM-ready prompt string.
    Chunks sorted by score descending — LLMs attend more to early context.
    Used by cli/rag_query.py and cli/app.py.
    """
    sorted_chunks = sorted(chunks, key=lambda rc: rc.score, reverse=True)
    context_blocks = []

    for i, rc in enumerate(sorted_chunks, 1):
        meta    = rc.chunk.metadata
        source  = meta.get("source", "unknown")
        page    = meta.get("page", "?")
        section = meta.get("section", "")
        header  = f"[{i}] Source: {source} | Page: {page}"
        if section:
            header += f" | Section: {section}"
        context_blocks.append(f"{header}\n{rc.chunk.content}")

    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def rag_node(state: AgentState, llm: LLMClientBase, retriever: HybridRetriever) -> AgentState:
    """
    LangGraph node: RAG.
    Reads:  state["query"], state["route"], state["retry_count"]
    Writes: state["retrieved_chunks"], state["context"], state["answer"], state["sources"]
    """
    query       = state["query"]
    retry_count = state.get("retry_count", 0)

    # ── Direct answer path ───────────────────────────────────────────────────
    if state.get("route") == "direct":
        logger.info(f"[RAG] Direct answer path for: '{query[:60]}'")
        answer = llm.generate(
            system_prompt=DIRECT_SYSTEM_PROMPT,
            user_prompt=query,
        )
        return {
            **state,
            "retrieved_chunks": [],
            "context":          "",
            "answer":           answer,
            "sources":          [],
            "grounded":         True,    # Direct answers skip critique
        }

    # ── RAG path ─────────────────────────────────────────────────────────────
    logger.info(f"[RAG] Retrieving for: '{query[:60]}' (attempt {retry_count + 1})")

    effective_query = query
    if retry_count > 0:
        effective_query = f"{query} (provide more detail and related context)"
        logger.info(f"[RAG] Retry — expanded query: '{effective_query[:80]}'")

    chunks = retriever.retrieve(effective_query)

    if not chunks:
        logger.warning("[RAG] No chunks retrieved — returning no-context answer")
        return {
            **state,
            "retrieved_chunks": [],
            "context":          "",
            "answer":           "I could not find relevant information in the knowledge base to answer your question.",
            "sources":          [],
            "grounded":         False,
        }

    context, sources = _format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    logger.info(f"[RAG] Generating answer with {len(chunks)} chunks")
    answer = llm.generate(system_prompt=RAG_SYSTEM_PROMPT, user_prompt=user_prompt)

    return {
        **state,
        "retrieved_chunks": chunks,
        "context":          context,
        "answer":           answer,
        "sources":          sources,
    }
