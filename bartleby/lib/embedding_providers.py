"""
Unified embedding interface supporting multiple providers.

Supports:
- sentence-transformers (local models)
- OpenAI embeddings API
- Ollama embeddings API
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
import os

from loguru import logger


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence-transformers provider."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded sentence-transformers model: {model_name} (dim={self.dimension})")

    def embed(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        tokens = self.model.tokenizer.tokenize(text)
        if len(tokens) > self.model.max_seq_length:
            logger.warning(f"Text has {len(tokens)} tokens, max is {self.model.max_seq_length}. Truncating.")
            # Truncate instead of raising error
            text = self.model.tokenizer.convert_tokens_to_string(
                tokens[:self.model.max_seq_length]
            )

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embedding.astype(np.float32, copy=False)

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_name(self) -> str:
        return self.model_name


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embeddings API provider."""

    # Model dimensions from OpenAI docs
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.dimension = self.MODEL_DIMENSIONS.get(model_name, 1536)
        logger.info(f"Initialized OpenAI embeddings: {model_name} (dim={self.dimension})")

    def embed(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return np.zeros(self.dimension, dtype=np.float32)

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_name(self) -> str:
        return self.model_name


class OllamaProvider(EmbeddingProvider):
    """Ollama embeddings API provider."""

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Requests package not installed. Install with: pip install requests"
            )

        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.dimension = None  # Will be detected on first embedding

        # Test connection and get dimension
        test_embedding = self._embed_internal("test")
        if test_embedding is not None:
            self.dimension = len(test_embedding)
            logger.info(f"Initialized Ollama embeddings: {model_name} (dim={self.dimension})")
        else:
            raise RuntimeError(
                f"Failed to initialize Ollama embedding model '{model_name}'. "
                f"Make sure Ollama is running at {base_url} and the model is pulled."
            )

    def _embed_internal(self, text: str) -> Optional[np.ndarray]:
        """Internal embedding method."""
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if "embedding" in data:
                embedding = np.array(data["embedding"], dtype=np.float32)

                # Normalize embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                return embedding
            else:
                logger.error(f"Unexpected Ollama response format: {data}")
                return None
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            return None

    def embed(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        embedding = self._embed_internal(text)
        if embedding is not None:
            return embedding
        else:
            return np.zeros(self.dimension, dtype=np.float32)

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_name(self) -> str:
        return self.model_name


def create_embedding_provider(
    provider: str = "sentence-transformers",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.

    Args:
        provider: Provider name - 'sentence-transformers', 'openai', or 'ollama'
        model: Model name (provider-specific)
        api_key: API key (for OpenAI)
        base_url: Base URL (for Ollama)

    Returns:
        EmbeddingProvider instance
    """
    provider = provider.lower()

    if provider == "sentence-transformers" or provider == "local":
        model = model or "BAAI/bge-base-en-v1.5"
        return SentenceTransformerProvider(model_name=model)

    elif provider == "openai":
        model = model or "text-embedding-3-small"
        return OpenAIProvider(model_name=model, api_key=api_key)

    elif provider == "ollama":
        model = model or "nomic-embed-text"
        base_url = base_url or "http://localhost:11434"
        return OllamaProvider(model_name=model, base_url=base_url)

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported: sentence-transformers, openai, ollama"
        )


# Backward compatibility wrapper
def embed_chunk(embedding_provider: EmbeddingProvider, body: str) -> np.ndarray:
    """
    Backward compatible wrapper for embedding generation.

    Args:
        embedding_provider: EmbeddingProvider instance
        body: Text to embed

    Returns:
        Embedding as numpy array
    """
    return embedding_provider.embed(body)
