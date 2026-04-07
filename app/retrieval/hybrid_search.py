"""
Hybrid Search

Combines dense vector search with sparse BM25 search for better retrieval.
"""

from typing import Any, Optional

try:
    from langchain_core.documents import Document
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


class HybridSearcher:
    """
    Hybrid search combining dense and sparse retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results
    from vector search and BM25.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            vector_store: Vector store for dense search
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 results
            rrf_k: RRF constant (higher = more weight to lower ranks)
        """
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        
        # Initialize BM25 index
        self.documents: list[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        
        self._build_bm25_index()
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from vector store documents."""
        logger.info("Building BM25 index")
        
        try:
            # Get all documents from vector store
            collection = self.vector_store.client.get_collection(
                self.vector_store.collection_name
            )
            results = collection.get(include=["documents", "metadatas"])
            
            if not results or not results.get("documents"):
                logger.warning("No documents found for BM25 index")
                return
            
            # Convert to Document objects
            self.documents = []
            for doc_text, metadata in zip(
                results["documents"],
                results["metadatas"] or [{}] * len(results["documents"])
            ):
                self.documents.append(Document(
                    page_content=doc_text,
                    metadata=metadata or {}
                ))
            
            # Tokenize for BM25
            tokenized = [
                self._tokenize(doc.page_content)
                for doc in self.documents
            ]
            
            self.bm25 = BM25Okapi(tokenized)
            
            logger.info(f"BM25 index built with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25 = None
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter (applied to vector search only)
        
        Returns:
            List of (document, score) tuples
        """
        # Get more results initially for fusion
        initial_k = min(k * 3, 50)
        
        # Vector search
        vector_results = self.vector_store.similarity_search(
            query=query,
            k=initial_k,
            filter=filter,
        )
        
        # BM25 search (if available)
        if self.bm25 and self.documents:
            bm25_results = self._bm25_search(query, initial_k)
        else:
            bm25_results = []
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            k=k,
        )
        
        return combined
    
    def _bm25_search(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """Perform BM25 search."""
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        
        results = [
            (self.documents[i], scores[i])
            for i in top_indices
            if scores[i] > 0
        ]
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
        k: int,
    ) -> list[tuple[Document, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each list
        """
        # Score dict: content hash -> (document, rrf_score)
        scores: dict[str, tuple[Document, float]] = {}
        
        # Process vector results
        for rank, (doc, _) in enumerate(vector_results):
            content_hash = hash(doc.page_content)
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)
            
            if content_hash in scores:
                existing_doc, existing_score = scores[content_hash]
                scores[content_hash] = (doc, existing_score + rrf_score)
            else:
                scores[content_hash] = (doc, rrf_score)
        
        # Process BM25 results
        for rank, (doc, _) in enumerate(bm25_results):
            content_hash = hash(doc.page_content)
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)
            
            if content_hash in scores:
                existing_doc, existing_score = scores[content_hash]
                scores[content_hash] = (doc, existing_score + rrf_score)
            else:
                scores[content_hash] = (doc, rrf_score)
        
        # Sort by combined score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:k]
    
    def refresh_index(self) -> None:
        """Refresh the BM25 index."""
        self._build_bm25_index()


# Alias for test compatibility
HybridSearch = HybridSearcher
