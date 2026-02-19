"""
mpeT (all-mpnet-base-v2) dense embedder.
Implements EmbedderBase — swap model in config/settings.py.
"""
import logging
from sentence_transformers import SentenceTransformer

from core.interfaces import EmbedderBase
from config.settings import config

logger = logging.getLogger(__name__)


class MpetEmbedder(EmbedderBase):
    """
    Dense embeddings via sentence-transformers (all-mpnet-base-v2).
    Model is downloaded once and cached locally by HuggingFace.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {config.embedding.model_name}")
        self.model = SentenceTransformer(
            config.embedding.model_name,
            device=config.embedding.device,
        )
        self.dimension = config.embedding.dimension
        logger.info(f"Embedder ready — dim={self.dimension}, device={config.embedding.device}")

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
