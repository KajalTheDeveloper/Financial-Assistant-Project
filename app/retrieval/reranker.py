"""
Reranker

Cross-encoder reranking for improved retrieval quality.
"""

from typing import Optional

try:
    from langchain_core.documents import Document
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranker for improved document relevance.
    
    Uses a cross-encoder model to score query-document pairs,
    which is more accurate than bi-encoder similarity.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 512,
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to run model on
            max_length: Maximum sequence length
        """
        self.model_name = model_name or settings.reranker_model
        self.device = device
        self.max_length = max_length
        
        logger.info("Initializing reranker", model=self.model_name)
        
        self.model = CrossEncoder(
            self.model_name,
            max_length=max_length,
            device=device,
        )
        
        logger.info("Reranker initialized")
    
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Number of top results to return
        
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        if top_k is None:
            top_k = len(documents)
        
        logger.debug(f"Reranking {len(documents)} documents")
        
        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        reranked = [doc for doc, score in doc_scores[:top_k]]
        
        logger.debug(f"Reranking complete, returning top {len(reranked)}")
        return reranked
    
    def rerank_with_scores(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        Rerank documents and return with scores.
        
        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of results
        
        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []
        
        if top_k is None:
            top_k = len(documents)
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.
        
        Args:
            query: Query text
            document: Document text
        
        Returns:
            Relevance score
        """
        return float(self.model.predict([[query, document]])[0])
