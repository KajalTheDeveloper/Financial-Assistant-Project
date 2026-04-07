"""
Vector Store

ChromaDB-based vector store for document storage and retrieval.
"""

from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
try:
    from langchain_core.documents import Document
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict
from langchain_chroma import Chroma

from app.core.config import settings
from app.core.exceptions import NoDocumentsError, VectorStoreError
from app.core.logging import get_logger
from app.retrieval.embeddings import EmbeddingGenerator, get_embeddings

logger = get_logger(__name__)


class VectorStore:
    """
    ChromaDB vector store wrapper.
    
    Provides:
    - Document storage with metadata
    - Similarity search
    - Filtering by metadata
    - Persistence
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            embedding_generator: Embedding generator instance
        """
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        
        # Create persist directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Get embedding generator
        self.embedding_generator = embedding_generator or get_embeddings()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Initialize LangChain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_generator.get_langchain_embeddings(),
        )
        
        logger.info(
            "Vector store initialized",
            collection=self.collection_name,
            persist_dir=self.persist_directory
        )
    
    def add_documents(
        self,
        documents: list[Document],
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            ids: Optional list of document IDs
        
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [
                    f"{doc.metadata.get('source_file', 'doc')}_{doc.metadata.get('chunk_index', i)}"
                    for i, doc in enumerate(documents)
                ]
            
            # Add to vector store
            added_ids = self.vectorstore.add_documents(
                documents=documents,
                ids=ids,
            )
            
            logger.info(f"Successfully added {len(added_ids)} documents")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        score_threshold: Optional[float] = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Metadata filter
            score_threshold: Minimum similarity score
        
        Returns:
            List of (document, score) tuples
        """
        try:
            # Search with scores
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter,
            )
            
            # Apply score threshold
            if score_threshold is not None:
                results = [(doc, score) for doc, score in results if score >= score_threshold]
            
            logger.debug(
                f"Search returned {len(results)} results",
                query=query[:50],
                k=k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}")
    
    def similarity_search_simple(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Simple similarity search returning only documents.
        
        Args:
            query: Query text
            k: Number of results
            filter: Metadata filter
        
        Returns:
            List of Document objects
        """
        results = self.similarity_search(query, k, filter)
        return [doc for doc, _ in results]
    
    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection."""
        collection = self.client.get_collection(self.collection_name)
        count = collection.count()
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
        }
    
    def list_sources(self) -> list[str]:
        """List unique source files in the collection."""
        collection = self.client.get_collection(self.collection_name)
        
        # Get all metadata
        results = collection.get(include=["metadatas"])
        
        if not results or not results.get("metadatas"):
            return []
        
        sources = set()
        for metadata in results["metadatas"]:
            if metadata and "source_file" in metadata:
                sources.add(metadata["source_file"])
        
        return sorted(list(sources))
    
    def delete_by_source(self, source_file: str) -> int:
        """
        Delete all chunks from a specific source file.
        
        Args:
            source_file: Name of the source file
        
        Returns:
            Number of documents deleted
        """
        collection = self.client.get_collection(self.collection_name)
        
        # Get IDs to delete
        results = collection.get(
            where={"source_file": source_file},
            include=["metadatas"]
        )
        
        if not results or not results.get("ids"):
            return 0
        
        ids_to_delete = results["ids"]
        collection.delete(ids=ids_to_delete)
        
        logger.info(f"Deleted {len(ids_to_delete)} chunks from {source_file}")
        return len(ids_to_delete)
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        logger.warning("Clearing all documents from vector store")
        self.client.delete_collection(self.collection_name)
        
        # Recreate empty collection
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_generator.get_langchain_embeddings(),
        )
    
    def is_empty(self) -> bool:
        """Check if the vector store is empty."""
        stats = self.get_collection_stats()
        return stats["document_count"] == 0
    
    def get_retriever(self, **kwargs):
        """Get a LangChain retriever interface."""
        return self.vectorstore.as_retriever(**kwargs)


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get the singleton vector store instance.
    
    Returns:
        VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def reset_vector_store() -> None:
    """Reset the vector store singleton."""
    global _vector_store
    if _vector_store is not None:
        _vector_store.clear()
    _vector_store = None
