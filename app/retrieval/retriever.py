"""
Document Retriever

Main retrieval interface combining vector search, hybrid search, and reranking.
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

from app.core.config import settings
from app.core.exceptions import NoDocumentsError, NoRelevantContextError
from app.core.logging import get_logger
from app.retrieval.vector_store import VectorStore, get_vector_store
from app.retrieval.reranker import Reranker
from app.retrieval.hybrid_search import HybridSearcher

logger = get_logger(__name__)


class DocumentRetriever:
    """
    Main document retrieval class.
    
    Features:
    - Vector similarity search
    - Optional hybrid search (vector + BM25)
    - Optional reranking
    - Metadata filtering
    - Configurable thresholds
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        enable_reranking: Optional[bool] = None,
        enable_hybrid_search: Optional[bool] = None,
        top_k: Optional[int] = None,
        initial_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store instance
            enable_reranking: Whether to use reranking
            enable_hybrid_search: Whether to use hybrid search
            top_k: Final number of results to return
            initial_k: Initial number of candidates to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.vector_store = vector_store or get_vector_store()
        self.enable_reranking = enable_reranking if enable_reranking is not None else settings.enable_reranking
        self.enable_hybrid_search = enable_hybrid_search if enable_hybrid_search is not None else settings.enable_hybrid_search
        self.top_k = top_k or settings.top_k_retrieval
        self.initial_k = initial_k or settings.initial_retrieval_k
        self.similarity_threshold = similarity_threshold or settings.similarity_threshold
        
        # Initialize optional components
        self.reranker = Reranker() if self.enable_reranking else None
        self.hybrid_searcher = None
        
        if self.enable_hybrid_search:
            self._initialize_hybrid_search()
    
    def _initialize_hybrid_search(self) -> None:
        """Initialize hybrid search with current documents."""
        try:
            # Get all documents from vector store
            stats = self.vector_store.get_collection_stats()
            if stats["document_count"] > 0:
                self.hybrid_searcher = HybridSearcher(self.vector_store)
                logger.info("Hybrid search initialized")
        except Exception as e:
            logger.warning(f"Could not initialize hybrid search: {e}")
            self.hybrid_searcher = None
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
        include_scores: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results (overrides default)
            filter: Metadata filter
            include_scores: Whether to include similarity scores
        
        Returns:
            List of dicts with document content, metadata, and scores
        """
        k = top_k or self.top_k
        
        # Check if vector store has documents
        if self.vector_store.is_empty():
            raise NoDocumentsError()
        
        logger.info(
            "Retrieving documents",
            query=query[:50],
            top_k=k,
            reranking=self.enable_reranking,
            hybrid=self.enable_hybrid_search
        )
        
        # Step 1: Initial retrieval
        if self.hybrid_searcher and self.enable_hybrid_search:
            # Use hybrid search
            candidates = self.hybrid_searcher.search(
                query=query,
                k=self.initial_k,
                filter=filter,
            )
        else:
            # Use vector search only
            candidates = self.vector_store.similarity_search(
                query=query,
                k=self.initial_k,
                filter=filter,
            )
        
        if not candidates:
            raise NoRelevantContextError(query, self.similarity_threshold)
        
        # Convert to documents if needed
        if isinstance(candidates[0], tuple):
            # (Document, score) tuples
            documents = [doc for doc, score in candidates]
            scores = [score for doc, score in candidates]
        else:
            documents = candidates
            scores = [1.0] * len(documents)
        
        # Step 2: Reranking
        if self.reranker and self.enable_reranking and len(documents) > k:
            reranked = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=k,
            )
            documents = reranked
            # Reranker provides its own scores
            scores = [1.0 - (i * 0.05) for i in range(len(documents))]  # Descending scores
        else:
            # Just take top k
            documents = documents[:k]
            scores = scores[:k]
        
        # Step 3: Apply similarity threshold
        results = []
        for doc, score in zip(documents, scores):
            if score >= self.similarity_threshold or not include_scores:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                if include_scores:
                    result["score"] = score
                results.append(result)
        
        if not results:
            raise NoRelevantContextError(query, self.similarity_threshold)
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def retrieve_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Retrieve documents as LangChain Document objects.
        
        Args:
            query: Search query
            top_k: Number of results
            filter: Metadata filter
        
        Returns:
            List of Document objects
        """
        results = self.retrieve(query, top_k, filter, include_scores=False)
        return [
            Document(page_content=r["content"], metadata=r["metadata"])
            for r in results
        ]
    
    def get_context_string(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
        include_metadata: bool = True,
    ) -> str:
        """
        Get retrieved context as a formatted string.
        
        Args:
            query: Search query
            top_k: Number of results
            filter: Metadata filter
            include_metadata: Whether to include source info
        
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k, filter)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            if include_metadata:
                source = result["metadata"].get("source_file", "Unknown")
                page = result["metadata"].get("page_number", "N/A")
                header = f"[Document {i}] Source: {source}, Page: {page}"
                context_parts.append(f"{header}\n{result['content']}")
            else:
                context_parts.append(f"[Document {i}]\n{result['content']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def refresh_hybrid_index(self) -> None:
        """Refresh hybrid search index after adding documents."""
        if self.enable_hybrid_search:
            self._initialize_hybrid_search()
