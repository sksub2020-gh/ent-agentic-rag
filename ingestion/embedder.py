"""
mpeT (all-mpnet-base-v2) dense embedder.
Implements EmbedderBase — swap model in config/settings.py.
"""
import logging
from typing import Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from core.interfaces import EmbedderBase
from config.settings import config

logger = logging.getLogger(__name__)


class MpetEmbedder(EmbedderBase):
    """
    Dense embeddings via sentence-transformers (all-mpnet-base-v2).
    Model is downloaded once and cached locally by HuggingFace.
    """
    _tokenizer = None

    @property
    def tokenizer(cls) -> Any:
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(config.embedding.model_name)
        return cls._tokenizer
    
    def __init__(self):
        logger.info(f"Loading embedding model: {config.embedding.model_name}")
        self.model = SentenceTransformer(
            config.embedding.model_name,
            device=config.embedding.device,
            local_files_only=True
        )
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedder ready — model={config.embedding.model_name} "
            f"dim={self.dimension} device={config.embedding.device}"
        )
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed a list of texts. Returns list of float vectors."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,   # Normalize for cosine similarity
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query — same model, just one text."""
        return self.embed([query])[0]
