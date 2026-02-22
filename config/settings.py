"""
Central config using pydantic-settings.

- All values have sensible defaults (works out of the box)
- Any value can be overridden via .env or environment variable
- Env var naming: <PREFIX>_<FIELD> e.g. LLM_MODEL, MILVUS_URI
- Type validation at startup — misconfigured vars fail loudly with clear errors

Add to requirements.txt: pydantic-settings>=2.0.0
"""
import os
from pathlib import Path
from typing import Annotated

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AfterValidator, Field, SecretStr

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Define the reusable logic
def resolve_abs_path(v: Path) -> Path:
    return v if v.is_absolute() else PROJECT_ROOT / v

# 2. Create a reusable type alias
AbsPath = Annotated[Path, AfterValidator(resolve_abs_path)]

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
    api_key:     SecretStr   = "ollama"
    model:       str   = "mistral:7b"
    temperature: float = 0.1
    max_tokens:  int   = 1024

    # model_config = {"env_prefix": "LLM_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class EmbeddingConfig(BaseSettings):
    model_name: str       = "sentence-transformers/all-mpnet-base-v2"
    dimension:  int | None = None
    device:     str       = "cpu"

    # model_config = {"env_prefix": "EMBEDDING_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class MilvusConfig(BaseSettings):
    uri:             AbsPath = "data/index/milvus_lite.db"
    collection_name: str = "rag_docs"
    metric_type:     str = "COSINE"

    # model_config = {"env_prefix": "MILVUS_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class BM25Config(BaseSettings):
    index_path: AbsPath = "data/index/bm25_index"
    method:     str = "lucene"

    # model_config = {"env_prefix": "BM25_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class RetrievalConfig(BaseSettings):
    top_k_dense:    int   = 20
    top_k_sparse:   int   = 20
    top_k_rerank:   int   = 5
    rrf_k:          int   = 60
    reranker_model: str   = "ms-marco-MiniLM-L-12-v2"

    # model_config = {"env_prefix": "RETRIEVAL_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class DoclingConfig(BaseSettings):
    max_tokens:        int  = 512
    min_tokens:        int  = 64
    overlap_tokens:    int  = 32
    supported_formats: list = Field(default=["pdf", "html", "docx"])
    ocr:               bool = False   # set DOCLING_OCR=true to enable EasyOCR


    # model_config = {"env_prefix": "DOCLING_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class SupabaseConfig(BaseSettings):
    connection_string: SecretStr = ""
    table_name:        str = "rag_chunks"

    # model_config = {"env_prefix": "SUPABASE_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

class QdrantConfig(BaseSettings):
    """
    Qdrant connection settings.

    Local file (default — no server needed):
        QDRANT_MODE=local
        QDRANT_PATH=data/index/qdrant

    In-memory (testing):
        QDRANT_MODE=memory

    Remote / Qdrant Cloud:
        QDRANT_MODE=remote
        QDRANT_URL=http://localhost:6333
        QDRANT_API_KEY=your-key   # only for Qdrant Cloud
    """
    mode:            str       = "local"            # local | memory | remote
    path:            AbsPath   = "data/index/qdrant"    # used when mode=local
    url:             str       = "http://localhost:6333"  # used when mode=remote
    api_key:         SecretStr = ""                 # Qdrant Cloud only
    collection_name: str       = "rag_chunks"
    dimension:       int | None = None              # auto-resolved from embedder

    model_config = {"env_prefix": "QDRANT_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

# Determine environment: .env.dev, .env.tst, or .env.prd
app_env = os.getenv("APP_ENV", "dev").lower()
env_file_path = f".env.{app_env}"

class AppConfig(BaseSettings):
    app_env:   str             = app_env
    # LLM 
    llm:       LLMConfig       = Field(default_factory=LLMConfig)

    # Parse, Chunk and Embeddding
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    docling:   DoclingConfig   = Field(default_factory=DoclingConfig)

    # Milvus + BM25s (local only)
    milvus:    MilvusConfig    = Field(default_factory=MilvusConfig)
    bm25:      BM25Config      = Field(default_factory=BM25Config)

    # qdrant + hybrid Search (local only)
    qdrant:    QdrantConfig    = Field(default_factory=QdrantConfig)

    # PgVector + hybrid Search (Supabase - cloud)
    supabase:  SupabaseConfig  = Field(default_factory=SupabaseConfig)
   
   # retrival
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    project_name: str          = "learn-agentic-rag"
    store_backend: str         = "supabase" # "supabase" | "milvus" | "qdrant"
    langchain_tracing_v2: str = False
    langchain_api_key: str = ''
    langchain_project: str = project_name

    # Environment variable loading
    model_config = SettingsConfigDict(
        env_file=env_file_path, 
        env_file_encoding='utf-8',
        env_nested_delimiter="__",
        extra='ignore'
    )

    # model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton — imported everywhere as: from config.settings import config
config = AppConfig()
