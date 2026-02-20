"""
Streamlit RAG Query App.
Run: streamlit run cli/app.py
"""
import logging
import streamlit as st
from config.settings import config

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("pymilvus").setLevel(logging.WARNING)

st.set_page_config(
    page_title="RAG Query",
    page_icon="🔍",
    layout="wide",
)

# ── Pipeline ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising pipeline...")
def load_pipeline():
    from core.llm_client import LLMClient
    from ingestion.embedder import MpetEmbedder
    from retrieval.hybrid_retriever import HybridRetriever, FlashRankReranker

    embedder  = MpetEmbedder()
    llm       = LLMClient()

    if config.store_backend == "supabase":
            from retrieval.supabase_store import SupabaseStore
            store = SupabaseStore()
            vector_store = sparse_store = store
    elif config.store_backend == "milvus":
        from retrieval.milvus_store import MilvusLiteStore
        from retrieval.bm25_store import BM25SStore
        vector_store = MilvusLiteStore(dimension=embedder.dimension)
        sparse_store = BM25SStore()
    else:
        st.warning(f"⚠️  Vector Backend {config.store_backend} is not implemented.")
        

    retriever = HybridRetriever(
        vector_store=vector_store,
        sparse_store=sparse_store,
        embedder=embedder,
        reranker=FlashRankReranker(),
    )
    return llm, retriever


@st.cache_data(show_spinner=False, ttl=30)
def check_ollama_status() -> bool:
    logger = logging.getLogger("core.llm_client")
    prev_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        from core.llm_client import LLMClient
        return LLMClient().health_check()
    except Exception:
        return False
    finally:
        logger.setLevel(prev_level)


def run_query(question: str, llm, retriever) -> dict:
    from agents.rag_node import RAG_SYSTEM_PROMPT, build_context_prompt

    chunks = retriever.retrieve(question)
    if not chunks:
        return {"answer": "No relevant context found in the knowledge base.", "chunks": []}

    answer = llm.generate(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=build_context_prompt(question, chunks),
    )
    return {
        "answer": answer,
        "chunks": [
            {
                "content": rc.chunk.content,
                "score":   round(rc.score, 4),
                "source":  rc.chunk.metadata.get("source", "unknown"),
                "page":    rc.chunk.metadata.get("page", "?"),
                "section": rc.chunk.metadata.get("section", ""),
            }
            for rc in chunks
        ],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_sources(chunks: list[dict], show_content: bool):
    with st.expander(f"📚 Sources ({len(chunks)} chunks)", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                src = chunk["source"].split("/")[-1]
                section = f" · *{chunk['section']}*" if chunk["section"] else ""
                st.markdown(f"**[{i}]** `{src}`{section}")
            with col2:
                st.caption(f"p. {chunk['page']}")
            with col3:
                st.caption(f"score {chunk['score']}")
            if show_content:
                st.code(
                    chunk["content"][:400] + ("..." if len(chunk["content"]) > 400 else ""),
                    language=None,
                )
            if i < len(chunks):
                st.divider()


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

    show_sources = st.toggle("Show sources", value=True)
    show_chunks  = st.toggle("Show chunk content", value=False)

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Ingest documents:**")
    st.code("python cli/ingestion_pipeline.py ./docs/", language="bash")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🔍 RAG Query")
# ybrid search · BM25 + Dense · FlashRank reranking · Mistral-7B")
_CAPTION = "BM25 + Dense" if config.store_backend == "milvus" else "SQL RRF"
st.caption(f"Hybrid search · {_CAPTION} · FlashRank · {config.llm.model.capitalize()} · {config.store_backend.capitalize()}")

try:
    llm, retriever = load_pipeline()
    pipeline_ready = True
except Exception as e:
    st.error(f"**Pipeline failed to load:** {e}")
    st.info("Make sure Ollama is running and documents are ingested.")
    pipeline_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("chunks"):
            render_sources(msg["chunks"], show_chunks)

if not st.session_state.messages:
    st.info("💬 Ask a question about your ingested documents to get started.")

if question := st.chat_input("Ask a question...", disabled=not pipeline_ready):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = run_query(question, llm, retriever)
        st.markdown(result["answer"])
        if show_sources and result["chunks"]:
            render_sources(result["chunks"], show_chunks)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "chunks":  result["chunks"],
    })
