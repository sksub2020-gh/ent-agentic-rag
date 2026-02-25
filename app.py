"""
Streamlit RAG Query App.
Run: streamlit run cli/app.py

Two modes toggled from sidebar:
  Linear   — HybridRetriever → FlashRank → LLM (fast, simple)
  Agentic  — Router → RAG → Critique → Output Guard (quality-gated)

LangSmith tracing: set LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true in .env
"""
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

for _lg in ["httpx","openai","sentence_transformers","pymilvus","qdrant_client","httpcore"]:
    logging.getLogger(_lg).setLevel(logging.WARNING)

st.set_page_config(
    page_title="DocRAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
# Strategy: use `color: inherit` everywhere text is displayed so it adapts
# to both light and dark Streamlit themes. Only structural/decorative elements
# use hardcoded colors. CSS media query overrides for dark mode where needed.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 900px; }

/* Mode badge — always dark bg with light text, fine on both themes */
.mode-badge {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 500;
    padding: 0.15rem 0.5rem; border-radius: 4px;
    background: #4f46e5; color: #ffffff; letter-spacing: 0.5px;
}

/* Source cards — inherit text color, use border for structure */
.source-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-left: 3px solid #4f46e5;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.83rem;
}
.source-card .src-title {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    color: inherit;
    font-size: 0.8rem;
}
.source-card .src-meta {
    opacity: 0.65;
    font-size: 0.75rem;
    margin-top: 0.1rem;
}
.source-card .src-content {
    font-size: 0.8rem;
    margin-top: 0.4rem;
    font-style: italic;
    opacity: 0.8;
    border-top: 1px solid rgba(128,128,128,0.2);
    padding-top: 0.35rem;
    line-height: 1.5;
}

/* Meta grid for agentic trace */
.meta-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.meta-card {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    text-align: center;
}
.meta-card .mc-label {
    font-size: 0.68rem;
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-family: 'IBM Plex Mono', monospace;
}
.meta-card .mc-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: inherit;
    font-family: 'IBM Plex Mono', monospace;
}

/* Reasoning / warning boxes — use amber/red tints that work on both themes */
.reasoning-box {
    border: 1px solid rgba(251,191,36,0.6);
    background: rgba(251,191,36,0.08);
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    font-size: 0.82rem;
    color: inherit;
    line-height: 1.5;
    margin-top: 0.5rem;
}
.warning-box {
    border: 1px solid rgba(249,115,22,0.6);
    background: rgba(249,115,22,0.08);
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    font-size: 0.82rem;
    color: inherit;
    margin-top: 0.4rem;
}
.blocked-box {
    border: 1px solid rgba(239,68,68,0.6);
    background: rgba(239,68,68,0.08);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: inherit;
    font-weight: 500;
    margin-top: 0.4rem;
}

/* Trace link */
.trace-link {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #6366f1;
}

/* Status pills */
.status-pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.2rem 0.6rem; border-radius: 100px;
    font-size: 0.72rem; font-weight: 500;
    font-family: 'IBM Plex Mono', monospace;
}
.status-ok  { background: #d1fae5; color: #065f46; }
.status-err { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising pipeline...")
def load_pipeline():
    from retrieval.store_factory import build_retriever
    from agents.graph import build_rag_graph
    from core.llm_client import LLMClient
    llm       = LLMClient()
    retriever = build_retriever(llm=llm)
    graph     = build_rag_graph(llm=llm, retriever=retriever)
    return llm, retriever, graph


@st.cache_data(show_spinner=False, ttl=30)
def check_ollama_status() -> bool:
    logging.getLogger("core.llm_client").setLevel(logging.CRITICAL)
    try:
        from core.llm_client import LLMClient
        return LLMClient().health_check()
    except Exception:
        return False


def get_latest_eval_scores() -> dict | None:
    import json
    from pathlib import Path
    results_dir = Path("evaluation/results")
    if not results_dir.exists():
        return None
    files = sorted(results_dir.glob("eval_*.json"), reverse=True)
    if not files:
        return None
    try:
        with open(files[0]) as f:
            data = json.load(f)
        return {
            "scores":    data.get("scores", {}),
            "passed":    data.get("passed", {}),
            "timestamp": data.get("timestamp", ""),
            "n_samples": data.get("n_samples", 0),
        }
    except Exception:
        return None


# ── Query runners ─────────────────────────────────────────────────────────────

def run_linear(question: str, llm, retriever) -> dict:
    from agents.rag_node import RAG_SYSTEM_PROMPT, build_context_prompt
    chunks = retriever.retrieve(question)
    if not chunks:
        return {"answer": "No relevant context found.", "chunks": [],
                "route": "rag", "grounded": None, "retries": 0, "blocked": False}
    answer = llm.generate(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=build_context_prompt(question, chunks),
    )
    return {"answer": answer, "chunks": _format_chunks(chunks),
            "route": "rag", "grounded": None, "retries": 0, "blocked": False}


def run_agentic(question: str, graph) -> dict:
    from agents.graph import run_query
    result = run_query(question, app=graph)
    chunks = [
        {"content": s.get("content", ""), "score": s.get("score", 0),
         "source": s.get("source", ""), "page": s.get("page", "?"),
         "section": s.get("section", "")}
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
        "run_id":           result.get("run_id"),
    }


def _format_chunks(chunks) -> list[dict]:
    return [
        {"content": rc.chunk.content, "score": round(rc.score, 4),
         "source": rc.chunk.metadata.get("source", "unknown"),
         "page": rc.chunk.metadata.get("page", "?"),
         "section": rc.chunk.metadata.get("section", "")}
        for rc in chunks
    ]


# ── Render helpers ────────────────────────────────────────────────────────────

def render_sources(chunks: list[dict], show_content: bool):
    if not chunks:
        return
    label = f"📚 Sources — {len(chunks)} chunk{'s' if len(chunks) != 1 else ''}"
    with st.expander(label, expanded=False):
        for i, chunk in enumerate(chunks, 1):
            src     = chunk.get("source", "").split("/")[-1] or "unknown"
            section = chunk.get("section", "")
            score   = chunk.get("score", 0)
            page    = chunk.get("page", "?")
            content = chunk.get("content", "")

            score_color = "#10b981" if score > 0.5 else "#f59e0b" if score > 0.2 else "#9ca3af"
            section_html = f" <span style='opacity:0.6'>· {section[:40]}</span>" if section else ""
            content_html = (
                f"<div class='src-content'>{content[:350]}{'…' if len(content) > 350 else ''}</div>"
                if show_content and content else ""
            )
            st.markdown(f"""
            <div class="source-card">
                <div class="src-title">[{i}] {src}{section_html}</div>
                <div class="src-meta">
                    Page {page} &nbsp;·&nbsp;
                    <span style="color:{score_color};font-weight:500">score {score:.4f}</span>
                </div>
                {content_html}
            </div>
            """, unsafe_allow_html=True)


def render_agentic_meta(result: dict, langsmith_project: str | None):
    with st.expander("🤖 Pipeline trace", expanded=False):
        if result.get("blocked"):
            st.markdown(
                f'<div class="blocked-box">🚫 Blocked: {result.get("block_reason","unknown")}</div>',
                unsafe_allow_html=True)
            return

        route        = result.get("route", "—")
        grounded     = result.get("grounded")
        retries      = result.get("retries", 0)
        grounded_str = "✅ yes" if grounded is True else "❌ no" if grounded is False else "—"
        route_str    = "🔍 rag" if route == "rag" else "💬 direct"

        st.markdown(f"""
        <div class="meta-grid">
            <div class="meta-card">
                <div class="mc-label">Route</div>
                <div class="mc-value">{route_str}</div>
            </div>
            <div class="meta-card">
                <div class="mc-label">Grounded</div>
                <div class="mc-value">{grounded_str}</div>
            </div>
            <div class="meta-card">
                <div class="mc-label">Retries</div>
                <div class="mc-value">{retries}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if result.get("router_reasoning"):
            st.markdown(
                f'<div class="reasoning-box"><strong>Router:</strong> {result["router_reasoning"]}</div>',
                unsafe_allow_html=True)
        if result.get("critique"):
            st.markdown(
                f'<div class="reasoning-box"><strong>Critique:</strong> {result["critique"]}</div>',
                unsafe_allow_html=True)
        for w in result.get("warnings", []):
            st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

        run_id = result.get("run_id")
        if run_id and langsmith_project:
            trace_url = f"https://smith.langchain.com/o/me/projects/{langsmith_project}/runs/{run_id}"
            st.markdown(
                f'<div style="margin-top:0.6rem">'
                f'<a class="trace-link" href="{trace_url}" target="_blank">🔗 View in LangSmith →</a>'
                f'</div>',
                unsafe_allow_html=True)


def render_eval_sidebar(eval_data: dict):
    """Eval scores with correctly colored bars — green=pass, red=fail."""
    scores = eval_data["scores"]
    passed = eval_data["passed"]
    ts     = eval_data["timestamp"]
    n      = eval_data["n_samples"]

    labels = {
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Ans. Relevancy",
        "context_precision": "Ctx. Precision",
        "context_recall":    "Ctx. Recall",
    }

    all_pass = all(passed.values())
    overall  = "✅ PASS" if all_pass else "❌ FAIL"
    st.caption(f"n={n} · {ts[:10]} · {overall}")

    for k, label in labels.items():
        s         = scores.get(k, 0)
        ok        = passed.get(k, False)
        icon      = "✅" if ok else "❌"
        bar_color = "#22c55e" if ok else "#ef4444"
        bar_pct   = int(s * 100)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption(f"{icon} {label}")
            st.markdown(
                f'<div style="background:rgba(128,128,128,0.2);border-radius:3px;height:5px;margin-bottom:6px">'
                f'<div style="width:{bar_pct}%;height:5px;border-radius:3px;background:{bar_color}"></div>'
                f'</div>',
                unsafe_allow_html=True)
        with c2:
            st.caption(f"{s:.3f}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Title — use native st.markdown heading, inherits theme color
    st.markdown("### 📄 DocRAG")

    ollama_ok  = check_ollama_status()
    status_cls = "status-ok" if ollama_ok else "status-err"
    status_txt = "Ollama connected" if ollama_ok else "Ollama unreachable"
    status_dot = "🟢" if ollama_ok else "🔴"
    st.markdown(
        f'<span class="status-pill {status_cls}">{status_dot} {status_txt}</span>',
        unsafe_allow_html=True)
    if not ollama_ok:
        st.caption("`ollama serve`")

    st.divider()
    st.caption("PIPELINE")
    mode = st.radio(
        "Mode", ["⚡ Linear", "🤖 Agentic"],
        label_visibility="collapsed",
        help="**Linear** — fast direct RAG\n\n**Agentic** — Router → RAG → Critique → Guard"
    )
    agentic_mode = mode == "🤖 Agentic"

    st.divider()
    st.caption("DISPLAY")
    show_sources = st.toggle("Sources", value=True)
    show_chunks  = st.toggle("Chunk content", value=False)
    show_trace   = st.toggle("Agent trace", value=True) if agentic_mode else False

    st.divider()
    st.caption("SESSION")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    eval_data = get_latest_eval_scores()
    if eval_data:
        st.divider()
        st.caption("LAST EVALUATION")
        render_eval_sidebar(eval_data)

    st.divider()
    st.caption("INGEST")
    st.code("python cli/ingestion_pipeline.py ./docs/", language="bash")


# ── Main ──────────────────────────────────────────────────────────────────────

from config.settings import config as _cfg

_langsmith_project = getattr(_cfg, "langsmith_project", None)
_backend_label     = {"supabase":"Supabase","qdrant":"Qdrant","milvus":"Milvus"}.get(
    _cfg.store_backend, _cfg.store_backend)
_mode_label        = "Agentic RAG" if agentic_mode else "Linear RAG"

# Header — native columns, no hardcoded colors
col_title, col_meta = st.columns([2, 5])
with col_title:
    st.markdown("### 📄 DocRAG")
with col_meta:
    st.markdown(
        f'<div style="padding-top:0.5rem;font-size:0.82rem;opacity:0.75">'
        f'<span class="mode-badge">{_mode_label}</span>'
        f'&nbsp;&nbsp;BGE-large · {_backend_label} hybrid · FlashRank · Mistral-7B'
        f'</div>',
        unsafe_allow_html=True)
st.divider()

# Load pipeline
pipeline_ready = False
pipeline_error = None
try:
    llm, retriever, graph = load_pipeline()
    pipeline_ready = True
except Exception as e:
    pipeline_error = str(e)

if pipeline_error:
    st.error("**Pipeline failed to load**")
    st.code(pipeline_error, language=None)
    st.caption("Check `.env` — valid `STORE_BACKEND`: `supabase`, `qdrant`, `milvus`")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            # mono font for questions, inherit color
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.9rem">'
                f'{msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            # plain st.markdown — fully theme-aware, no color overrides
            st.markdown(msg["content"])
            if show_sources and msg.get("chunks"):
                render_sources(msg["chunks"], show_chunks)
            if agentic_mode and show_trace and msg.get("agentic_meta"):
                render_agentic_meta(msg["agentic_meta"], _langsmith_project)

# Empty state
if not st.session_state.messages and pipeline_ready:
    st.markdown(
        """
        <div style="text-align:center;padding:3.5rem 1rem;opacity:0.55">
            <div style="font-size:2.5rem;margin-bottom:0.75rem">📄</div>
            <div style="font-size:1rem;font-weight:500;margin-bottom:0.35rem">
                Ask anything about your documents
            </div>
            <div style="font-size:0.82rem">
                Hybrid search · BM42 sparse + BGE-large dense · FlashRank reranking · HyDE
            </div>
        </div>
        """,
        unsafe_allow_html=True)

# New input
if question := st.chat_input("Ask a question...", disabled=not pipeline_ready):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.9rem">'
            f'{question}</div>',
            unsafe_allow_html=True)

    with st.chat_message("assistant"):
        spinner_msg = "Running agentic pipeline..." if agentic_mode else "Retrieving and generating..."
        with st.spinner(spinner_msg):
            result = run_agentic(question, graph) if agentic_mode else run_linear(question, llm, retriever)

        # Native st.markdown — adapts to light/dark automatically
        st.markdown(result["answer"])

        if show_sources and result.get("chunks"):
            render_sources(result["chunks"], show_chunks)
        if agentic_mode and show_trace:
            render_agentic_meta(result, _langsmith_project)

    st.session_state.messages.append({
        "role":         "assistant",
        "content":      result["answer"],
        "chunks":       result.get("chunks", []),
        "agentic_meta": result if agentic_mode else None,
    })
