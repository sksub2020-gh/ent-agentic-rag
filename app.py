"""
Streamlit RAG Query App.
Run: streamlit run cli/app.py

Two modes toggled from sidebar:
  Linear   — HybridRetriever → FlashRank → LLM (fast, simple)
  Agentic  — Router → RAG → Critique → Output Guard (quality-gated)
"""
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("pymilvus").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

st.set_page_config(
    page_title="RAG Query",
    page_icon="🔍",
    layout="wide",
)


# ── Pipeline ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising pipeline...")
def load_pipeline():
    """
    Builds both pipelines once — cached for session lifetime.
    Returns (llm, retriever, graph) where graph is the compiled LangGraph app.
    """
    from retrieval.store_factory import build_pipeline
    from agents.graph import build_rag_graph

    llm, retriever = build_pipeline()
    graph = build_rag_graph(llm=llm, retriever=retriever)
    return llm, retriever, graph


@st.cache_data(show_spinner=False, ttl=30)
def check_ollama_status() -> bool:
    logger = logging.getLogger("core.llm_client")
    prev = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        from core.llm_client import LLMClient
        return LLMClient().health_check()
    except Exception:
        return False
    finally:
        logger.setLevel(prev)


# ── Query runners ─────────────────────────────────────────────────────────────

def run_linear(question: str, llm, retriever) -> dict:
    """Simple RAG — no guardrails, no critique, no retry."""
    from agents.rag_node import RAG_SYSTEM_PROMPT, build_context_prompt

    chunks = retriever.retrieve(question)
    if not chunks:
        return {
            "answer":  "No relevant context found in the knowledge base.",
            "chunks":  [],
            "route":   "rag",
            "grounded": None,
            "retries":  0,
            "blocked":  False,
        }

    answer = llm.generate(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=build_context_prompt(question, chunks),
    )
    return {
        "answer":  answer,
        "chunks":  _format_chunks(chunks),
        "route":   "rag",
        "grounded": None,   # not evaluated in linear mode
        "retries":  0,
        "blocked":  False,
    }


def run_agentic(question: str, graph) -> dict:
    """Full agentic RAG — Router → RAG → Critique → Output Guard."""
    from agents.graph import run_query
    result = run_query(question, app=graph)

    # Normalise chunks from AgentState sources list
    chunks = [
        {
            "content": "",   # sources list has no content — show score/page only
            "score":   s.get("score", 0),
            "source":  s.get("source", ""),
            "page":    s.get("page", "?"),
            "section": s.get("section", ""),
        }
        for s in result.get("sources", [])
    ]
    return {
        "answer":           result["answer"],
        "chunks":           chunks,
        "route":            result.get("route", ""),
        "router_reasoning": result.get("router_reasoning", ""),
        "grounded":         result.get("grounded", False),
        "critique":         result.get("critique_reasoning", ""),
        "retries":          result.get("retry_count", 0),
        "blocked":          result.get("blocked", False),
        "block_reason":     result.get("block_reason"),
        "warnings":         result.get("guard_warnings", []),
    }


def _format_chunks(chunks) -> list[dict]:
    return [
        {
            "content": rc.chunk.content,
            "score":   round(rc.score, 4),
            "source":  rc.chunk.metadata.get("source", "unknown"),
            "page":    rc.chunk.metadata.get("page", "?"),
            "section": rc.chunk.metadata.get("section", ""),
        }
        for rc in chunks
    ]


# ── Render helpers ────────────────────────────────────────────────────────────

def render_sources(chunks: list[dict], show_content: bool):
    if not chunks:
        return
    with st.expander(f"📚 Sources ({len(chunks)} chunks)", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                src = chunk["source"].split("/")[-1]
                section = f" · *{chunk['section']}*" if chunk.get("section") else ""
                st.markdown(f"**[{i}]** `{src}`{section}")
            with col2:
                st.caption(f"p. {chunk['page']}")
            with col3:
                st.caption(f"score {chunk['score']}")
            if show_content and chunk.get("content"):
                st.code(
                    chunk["content"][:400] + ("..." if len(chunk["content"]) > 400 else ""),
                    language=None,
                )
            if i < len(chunks):
                st.divider()


def render_agentic_meta(result: dict):
    """Show agentic pipeline metadata — route, grounding, retries, warnings."""
    with st.expander("🤖 Agent trace", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            route = result.get("route", "")
            icon  = "🔍" if route == "rag" else "💬"
            st.metric("Route", f"{icon} {route}")

        with col2:
            grounded = result.get("grounded")
            if grounded is True:
                st.metric("Grounded", "✅ Yes")
            elif grounded is False:
                st.metric("Grounded", "❌ No")
            else:
                st.metric("Grounded", "—")

        with col3:
            st.metric("Retries", result.get("retries", 0))

        if result.get("router_reasoning"):
            st.caption(f"**Router:** {result['router_reasoning']}")

        if result.get("critique"):
            st.caption(f"**Critique:** {result['critique']}")

        if result.get("blocked"):
            st.error(f"🚫 Blocked: {result.get('block_reason', 'unknown reason')}")

        if result.get("warnings"):
            for w in result["warnings"]:
                st.warning(f"⚠️ {w}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    ollama_ok = check_ollama_status()
    if ollama_ok:
        st.success("🟢 Ollama connected")
    else:
        st.error("🔴 Ollama unreachable")
        st.caption("Run: `ollama serve`")

    st.divider()

    # Mode toggle
    mode = st.radio(
        "Pipeline mode",
        ["⚡ Linear RAG", "🤖 Agentic RAG"],
        help=(
            "**Linear** — fast, direct retrieval + generation\n\n"
            "**Agentic** — Router → RAG → Critique → Guard (slower, quality-gated)"
        ),
    )
    agentic_mode = mode == "🤖 Agentic RAG"

    st.divider()

    show_sources  = st.toggle("Show sources", value=True)
    show_chunks   = st.toggle("Show chunk content", value=False)
    show_trace    = st.toggle("Show agent trace", value=True) if agentic_mode else False

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Ingest documents:**")
    st.code("python cli/ingestion_pipeline.py ./docs/", language="bash")


# ── Main ──────────────────────────────────────────────────────────────────────

from config.settings import config as _config
_backend_label = {
    "supabase": "Supabase",
    "qdrant":   "Qdrant",
    "milvus":   "Milvus",
}.get(_config.store_backend, f"Unknown ({_config.store_backend})")

st.title("🔍 RAG Query")
st.caption(
    f"{'🤖 Agentic' if agentic_mode else '⚡ Linear'} · "
    f"Hybrid search · FlashRank · Mistral-7B · {_backend_label}"
)

pipeline_ready = False
pipeline_error = None
try:
    llm, retriever, graph = load_pipeline()
    pipeline_ready = True
except Exception as e:
    pipeline_error = str(e)

# Show pipeline error in a clean formatted way
if pipeline_error:
    st.error("**Pipeline failed to load**")
    st.code(pipeline_error, language=None)
    st.caption("Check your `.env` — valid options for `STORE_BACKEND`: `supabase`, `qdrant`, `milvus`")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.code(msg["content"], language=None)
        else:
            st.markdown(msg["content"])
            if show_sources and msg.get("chunks"):
                render_sources(msg["chunks"], show_chunks)
            if agentic_mode and show_trace and msg.get("agentic_meta"):
                render_agentic_meta(msg["agentic_meta"])

# Ready / error state message
if not st.session_state.messages:
    if pipeline_ready:
        st.info(
            f"💬 {'Agentic' if agentic_mode else 'Linear'} RAG ready — "
            "ask a question about your ingested documents."
        )
    # If pipeline failed, error is already shown above — no duplicate message

# Handle new input
if question := st.chat_input("Ask a question...", disabled=not pipeline_ready):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        spinner_msg = "Running agentic pipeline..." if agentic_mode else "Retrieving and generating..."
        with st.spinner(spinner_msg):
            if agentic_mode:
                result = run_agentic(question, graph)
            else:
                result = run_linear(question, llm, retriever)

        st.markdown(result["answer"])

        if show_sources and result.get("chunks"):
            render_sources(result["chunks"], show_chunks)

        if agentic_mode and show_trace:
            render_agentic_meta(result)

    st.session_state.messages.append({
        "role":         "assistant",
        "content":      result["answer"],
        "chunks":       result.get("chunks", []),
        "agentic_meta": result if agentic_mode else None,
    })
