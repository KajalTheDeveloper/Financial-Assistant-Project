"""
Embedding Generator

Generate dense vector embeddings for documents and queries.
"""

from functools import lru_cache
from typing import Optional, Union

try:
    from langchain_core.documents import Document
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings using configurable models.
    
    Supports:
    - HuggingFace models (BGE, all-MiniLM, etc.)
    - OpenAI embeddings (optional)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        normalize: bool = True,
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the embedding model
            device: Device to run model on (cpu/cuda)
            normalize: Whether to normalize embeddings
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device
        self.normalize = normalize
        
        logger.info("Initializing embedding model", model=self.model_name)
        
        # Model-specific kwargs
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": normalize}
        
        # Add batch size for efficiency
        encode_kwargs["batch_size"] = 32
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        
        # Get embedding dimension
        sample = self.embeddings.embed_query("test")
        self.dimension = len(sample)
        
        logger.info(
            "Embedding model initialized",
            model=self.model_name,
            dimension=self.dimension
        )
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.debug(f"Embedding {len(texts)} documents")
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a single query.
        
        Some models have separate query embeddings (e.g., BGE).
        
        Args:
            query: Query text to embed
        
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(query)
    
    def embed_document_objects(
        self,
        documents: list[Document]
    ) -> list[tuple[Document, list[float]]]:
        """
        Generate embeddings for Document objects.
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of (document, embedding) tuples
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.embed_documents(texts)
        return list(zip(documents, embeddings))
    
    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Get the underlying LangChain embeddings object."""
        return self.embeddings


@lru_cache(maxsize=1)
def get_embeddings(model_name: Optional[str] = None) -> EmbeddingGenerator:
    """
    Get cached embedding generator.
    
    Args:
        model_name: Optional model name override
    
    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(model_name=model_name)


# Alias for test compatibility
EmbeddingModel = EmbeddingGenerator
