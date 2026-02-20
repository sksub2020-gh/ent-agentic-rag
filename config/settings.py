"""
Central config using pydantic-settings.

- All values have sensible defaults (works out of the box)
- Any value can be overridden via .env or environment variable
- Env var naming: <PREFIX>_<FIELD> e.g. LLM_MODEL, MILVUS_URI
- Type validation at startup — misconfigured vars fail loudly with clear errors

Add to requirements.txt: pydantic-settings>=2.0.0
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class LLMConfig(BaseSettings):
    """
    Universal OpenAI-compatible LLM config.
    Provider is just a label — the client uses base_url + api_key + model.

    Switching providers = changing .env values only, zero code changes:
      ollama  → base_url=http://localhost:11434/v1  api_key=ollama
      openai  → base_url=https://api.openai.com/v1  api_key=sk-...
      groq    → base_url=https://api.groq.com/openai/v1  api_key=gsk_...
    """
    provider:    str   = "ollama"
    base_url:    str   = "http://localhost:11434/v1"
    api_key:     str   = "ollama"
    model:       str   = "mistral:7b"
    temperature: float = 0.1
    max_tokens:  int   = 1024

    model_config = {"env_prefix": "LLM_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class EmbeddingConfig(BaseSettings):
    model_name: str       = "sentence-transformers/all-mpnet-base-v2"
    dimension:  int | None = None
    device:     str       = "cpu"

    model_config = {"env_prefix": "EMBEDDING_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class MilvusConfig(BaseSettings):
    uri:             str = "./data/index/milvus_lite.db"
    collection_name: str = "rag_docs"
    metric_type:     str = "COSINE"

    model_config = {"env_prefix": "MILVUS_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class BM25Config(BaseSettings):
    index_path: str = "./data/index/bm25_index"
    method:     str = "lucene"

    model_config = {"env_prefix": "BM25_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class RetrievalConfig(BaseSettings):
    top_k_dense:    int   = 20
    top_k_sparse:   int   = 20
    top_k_rerank:   int   = 5
    rrf_k:          int   = 60
    reranker_model: str   = "ms-marco-MiniLM-L-12-v2"

    model_config = {"env_prefix": "RETRIEVAL_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class DoclingConfig(BaseSettings):
    max_tokens:        int  = 512
    min_tokens:        int  = 64
    overlap_tokens:    int  = 32
    supported_formats: list = Field(default=["pdf", "html", "docx"])

    model_config = {"env_prefix": "DOCLING_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class SupabaseConfig(BaseSettings):
    connection_string: str = ""
    table_name:        str = "rag_chunks"

    model_config = {"env_prefix": "SUPABASE_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}




class AppConfig(BaseSettings):
    llm:       LLMConfig       = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    milvus:    MilvusConfig    = Field(default_factory=MilvusConfig)
    bm25:      BM25Config      = Field(default_factory=BM25Config)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    docling:   DoclingConfig   = Field(default_factory=DoclingConfig)
    supabase:  SupabaseConfig  = Field(default_factory=SupabaseConfig)
    project_name: str          = "learn-agentic-rag"
    store_backend: str         = "supabase" # "supabase" | "milvus"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — imported everywhere as: from config.settings import config
config = AppConfig()
