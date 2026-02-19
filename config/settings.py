"""
Central config — change here, works everywhere.
"""

from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"  # Required by SDK, ignored by Ollama
    model: str = "mistral:7b"
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"  # mpeT
    device: str = "cpu"  # Switch to "cuda" if GPU available


@dataclass
class MilvusConfig:
    uri: str = "./data/index/milvus_lite.db"  # Local file-based Milvus-Lite
    collection_name: str = "rag_docs"
    metric_type: str = "COSINE"


@dataclass
class BM25Config:
    index_path: str = "./data/index/bm25_index"
    method: str = "lucene"  # lucene | bm25+ | robertson


@dataclass
class RetrievalConfig:
    top_k_dense: int = 20  # Fetch more, rerank down
    top_k_sparse: int = 20
    top_k_rerank: int = 5  # Final chunks sent to LLM
    rrf_k: int = 60  # RRF constant (standard = 60)
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"  # FlashRank model


@dataclass
class DoclingConfig:
    # Hybrid chunker settings
    max_tokens: int = 512
    min_tokens: int = 64
    overlap_tokens: int = 32
    supported_formats: list = field(default_factory=lambda: ["pdf", "html", "docx"])


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    docling: DoclingConfig = field(default_factory=DoclingConfig)
    project_name: str = "learn-agentic-rag"


# Singleton config instance
config = AppConfig()
