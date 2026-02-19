"""
Agent node unit tests — each node tested in isolation with mocks.
No Ollama, no Milvus, no BM25S needed to run these.

Run: python -m pytest agents/tests/ -v
"""
import pytest
from unittest.mock import MagicMock

from agents.state import AgentState
from agents.router_node import router_node
from agents.rag_node import rag_node
from agents.critique_node import critique_node, should_retry
from core.interfaces import RetrievedChunk, Chunk


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_state(**overrides) -> AgentState:
    base: AgentState = {
        "query": "What is the refund policy?",
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
    return {**base, **overrides}


def make_chunk(content: str, chunk_id: str = "c001") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="doc001",
            content=content,
            metadata={"source": "policy.pdf", "page": 2},
        ),
        score=0.92,
        source="reranked",
    )


def mock_llm(response: str) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = response
    return llm


# ── Router Node Tests ─────────────────────────────────────────────────────────

class TestRouterNode:

    def test_routes_to_rag_for_document_query(self):
        llm = mock_llm('{"route": "rag", "reasoning": "Requires document lookup"}')
        state = make_state(query="What is the refund policy?")
        result = router_node(state, llm=llm)
        assert result["route"] == "rag"
        assert result["router_reasoning"] == "Requires document lookup"

    def test_routes_to_direct_for_general_query(self):
        llm = mock_llm('{"route": "direct", "reasoning": "General knowledge question"}')
        state = make_state(query="What is Python?")
        result = router_node(state, llm=llm)
        assert result["route"] == "direct"

    def test_fallback_to_rag_on_bad_json(self):
        llm = mock_llm("I cannot decide")       # Not valid JSON
        state = make_state(query="Test?")
        result = router_node(state, llm=llm)
        assert result["route"] == "rag"         # Safe fallback

    def test_fallback_to_rag_on_unexpected_route(self):
        llm = mock_llm('{"route": "unknown_value", "reasoning": "???"}')
        state = make_state(query="Test?")
        result = router_node(state, llm=llm)
        assert result["route"] == "rag"


# ── RAG Node Tests ────────────────────────────────────────────────────────────

class TestRagNode:

    def test_direct_path_skips_retrieval(self):
        llm = mock_llm("Python is a programming language.")
        mock_retriever = MagicMock()
        state = make_state(query="What is Python?", route="direct")

        result = rag_node(state, llm=llm, retriever=mock_retriever)

        mock_retriever.retrieve.assert_not_called()
        assert result["answer"] == "Python is a programming language."
        assert result["grounded"] is True      # Direct answers bypass critique

    def test_rag_path_calls_retriever(self):
        chunks = [make_chunk("Refund within 30 days with receipt.")]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = chunks
        llm = mock_llm("You can get a refund within 30 days.")
        state = make_state(query="What is the refund policy?", route="rag")

        result = rag_node(state, llm=llm, retriever=mock_retriever)

        mock_retriever.retrieve.assert_called_once()
        assert result["answer"] == "You can get a refund within 30 days."
        assert len(result["retrieved_chunks"]) == 1
        assert len(result["sources"]) == 1

    def test_no_chunks_returns_no_context_answer(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        llm = mock_llm("Irrelevant")
        state = make_state(query="?", route="rag")

        result = rag_node(state, llm=llm, retriever=mock_retriever)

        assert "could not find" in result["answer"].lower()
        assert result["grounded"] is False

    def test_retry_expands_query(self):
        chunks = [make_chunk("Some content")]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = chunks
        llm = mock_llm("Answer")
        state = make_state(query="What is X?", route="rag", retry_count=1)

        rag_node(state, llm=llm, retriever=mock_retriever)

        # On retry, query should be expanded
        call_args = mock_retriever.retrieve.call_args[0][0]
        assert "more detail" in call_args


# ── Critique Node Tests ───────────────────────────────────────────────────────

class TestCritiqueNode:

    def test_passes_grounded_answer(self):
        llm = mock_llm('{"grounded": true, "reasoning": "Answer matches context", "issues": []}')
        state = make_state(
            route="rag",
            context="Refund within 30 days.",
            answer="You can get a refund in 30 days.",
        )
        result = critique_node(state, llm=llm)
        assert result["grounded"] is True

    def test_fails_hallucinated_answer(self):
        llm = mock_llm('{"grounded": false, "reasoning": "Claim not in context", "issues": ["60 days not mentioned"]}')
        state = make_state(
            route="rag",
            context="Refund within 30 days.",
            answer="You can get a refund in 60 days.",
        )
        result = critique_node(state, llm=llm)
        assert result["grounded"] is False
        assert result["retry_count"] == 1

    def test_skips_critique_for_direct_route(self):
        llm = mock_llm("irrelevant")
        state = make_state(route="direct", grounded=True)
        result = critique_node(state, llm=llm)
        assert result["grounded"] is True
        llm.generate.assert_not_called()

    def test_max_retries_appends_disclaimer(self):
        llm = mock_llm('{"grounded": false, "reasoning": "Still not grounded", "issues": []}')
        state = make_state(
            route="rag",
            context="Some context.",
            answer="Ungrounded answer.",
            retry_count=2,              # Already at MAX_RETRIES
        )
        result = critique_node(state, llm=llm)
        assert result["grounded"] is True       # Force exits loop
        assert "⚠️" in result["answer"]         # Disclaimer added

    def test_fallback_on_bad_json(self):
        llm = mock_llm("Not JSON at all")
        state = make_state(route="rag", context="ctx", answer="ans")
        result = critique_node(state, llm=llm)
        assert result["grounded"] is False      # Safe fallback


# ── should_retry Edge Tests ───────────────────────────────────────────────────

class TestShouldRetry:

    def test_ends_when_grounded(self):
        state = make_state(grounded=True, retry_count=0)
        assert should_retry(state) == "end"

    def test_retries_when_not_grounded_and_under_limit(self):
        state = make_state(grounded=False, retry_count=1)
        assert should_retry(state) == "retry"

    def test_ends_when_max_retries_exceeded(self):
        state = make_state(grounded=False, retry_count=3)
        assert should_retry(state) == "end"
