"""
Abstract base interfaces — the plug-n-play contracts.
To swap a component (e.g. Milvus → Pinecone), implement the relevant interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A raw document before chunking."""
    doc_id: str
    content: str
    source: str                          # file path or URL
    doc_type: str = "text"               # pdf | html | table | image
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A processed chunk ready for embedding and indexing."""
    chunk_id: str
    doc_id: str
    content: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # page, section, type, etc.


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with its score."""
    chunk: Chunk
    score: float
    source: str = ""                     # "dense" | "sparse" | "reranked"


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class ChunkerBase(ABC):
    """Implement this to swap Docling for another chunker."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        ...


class EmbedderBase(ABC):
    """Implement this to swap mpeT for another embedding model."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return dense embeddings for a list of texts."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        ...


class VectorStoreBase(ABC):
    """
    Implement this to swap Milvus-Lite for Pinecone, Qdrant, Weaviate, etc.
    Only 4 methods to implement — that's the whole contract.
    """

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks with their embeddings."""
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return top_k most similar chunks."""
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks stored."""
        ...


class SparseStoreBase(ABC):
    """Implement this to swap BM25S for Elasticsearch, OpenSearch, etc."""

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Build/update sparse index from chunks."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return top_k chunks by BM25 score."""
        ...


class RerankerBase(ABC):
    """Implement this to swap FlashRank for Cohere, BGE, etc."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks and return top_k."""
        ...


class LLMClientBase(ABC):
    """Implement this to swap Ollama for OpenAI, Anthropic, etc."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given prompts."""
        ...
