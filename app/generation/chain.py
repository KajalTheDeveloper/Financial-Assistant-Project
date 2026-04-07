"""
RAG Chain

Complete RAG pipeline combining retrieval and generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Optional

from app.core.config import settings
from app.core.exceptions import NoDocumentsError, NoRelevantContextError
from app.core.logging import get_logger
from app.retrieval.retriever import DocumentRetriever
from app.retrieval.vector_store import get_vector_store
from app.generation.llm import LLMGenerator, get_llm
from app.generation.prompts import PromptTemplates, SYSTEM_PROMPT
from app.generation.response_formatter import ResponseFormatter

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Structured RAG response with metadata."""
    answer: str
    sources: list[dict[str, Any]]
    query: str
    rewritten_query: Optional[str] = None
    confidence: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = ""
    num_sources: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "model": self.model,
            "num_sources": self.num_sources,
        }


class RAGChain:
    """
    Complete RAG pipeline.
    
    Features:
    - Query preprocessing/rewriting
    - Multi-stage retrieval
    - Context assembly
    - LLM generation with streaming
    - Response formatting with citations
    - Confidence estimation
    """
    
    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        llm: Optional[LLMGenerator] = None,
        enable_query_rewriting: Optional[bool] = None,
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: Document retriever
            llm: LLM generator
            enable_query_rewriting: Whether to rewrite queries
        """
        self.retriever = retriever or DocumentRetriever()
        self.llm = llm or get_llm()
        self.enable_query_rewriting = (
            enable_query_rewriting 
            if enable_query_rewriting is not None 
            else settings.enable_query_rewriting
        )
        self.formatter = ResponseFormatter()
        
        logger.info("RAG chain initialized")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
        stream: bool = False,
    ) -> RAGResponse | Iterator[str]:
        """
        Execute RAG query.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            filter: Metadata filter for retrieval
            stream: Whether to stream the response
        
        Returns:
            RAGResponse or Iterator for streaming
        """
        import time
        start_time = time.time()
        
        logger.info("Processing RAG query", question=question[:50])
        
        # Step 1: Query preprocessing
        rewritten_query = None
        if self.enable_query_rewriting:
            rewritten_query = self._rewrite_query(question)
            search_query = rewritten_query
        else:
            search_query = question
        
        # Step 2: Retrieve relevant documents
        try:
            retrieved = self.retriever.retrieve(
                query=search_query,
                top_k=top_k,
                filter=filter,
            )
        except NoDocumentsError:
            return self._create_no_documents_response(question)
        except NoRelevantContextError:
            return self._create_no_context_response(question)
        
        # Step 3: Build context
        context = self._build_context(retrieved)
        sources = self._extract_sources(retrieved)
        
        # Step 4: Generate response
        prompt = PromptTemplates.format_answer_prompt(context, question)
        
        if stream:
            return self._stream_response(
                prompt=prompt,
                question=question,
                rewritten_query=rewritten_query,
                sources=sources,
                start_time=start_time,
            )
        else:
            answer = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            confidence = self._estimate_confidence(retrieved, answer)
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                query=question,
                rewritten_query=rewritten_query,
                confidence=confidence,
                latency_ms=latency_ms,
                model=self.llm.model_name,
                num_sources=len(sources),
            )
    
    def _rewrite_query(self, query: str) -> str:
        """Rewrite query for better retrieval."""
        try:
            prompt = PromptTemplates.format_query_rewrite(query)
            rewritten = self.llm.generate(prompt)
            logger.debug(f"Query rewritten: {query[:30]} -> {rewritten[:30]}")
            return rewritten.strip()
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return query
    
    def _build_context(self, retrieved: list[dict]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(retrieved, 1):
            source = doc["metadata"].get("source_file", "Unknown")
            page = doc["metadata"].get("page_number", "N/A")
            
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {source}, Page: {page}\n"
                f"Content:\n{doc['content']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_sources(self, retrieved: list[dict]) -> list[dict]:
        """Extract source information from retrieved documents."""
        sources = []
        
        for i, doc in enumerate(retrieved, 1):
            metadata = doc["metadata"]
            sources.append({
                "index": i,
                "source_file": metadata.get("source_file", "Unknown"),
                "page_number": metadata.get("page_number", "N/A"),
                "score": doc.get("score", 0.0),
                "chunk_type": metadata.get("chunk_type", "text"),
                "preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
            })
        
        return sources
    
    def _estimate_confidence(
        self,
        retrieved: list[dict],
        answer: str
    ) -> str:
        """Estimate response confidence based on retrieval quality."""
        if not retrieved:
            return "LOW"
        
        # Check retrieval scores
        scores = [doc.get("score", 0) for doc in retrieved]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Check source diversity
        sources = set(doc["metadata"].get("source_file") for doc in retrieved)
        
        # Determine confidence
        if avg_score > 0.8 and len(sources) >= 2:
            return "HIGH"
        elif avg_score > 0.6 or len(sources) >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _stream_response(
        self,
        prompt: str,
        question: str,
        rewritten_query: Optional[str],
        sources: list[dict],
        start_time: float,
    ) -> Iterator[str]:
        """Stream the response with metadata at the end."""
        full_response = ""
        
        for chunk in self.llm.generate_stream(prompt, system_prompt=SYSTEM_PROMPT):
            full_response += chunk
            yield chunk
        
        # After streaming, you can access the full response
        # The caller can construct the RAGResponse if needed
    
    def _create_no_documents_response(self, question: str) -> RAGResponse:
        """Create response when no documents are available."""
        answer = (
            "I don't have any documents to search through yet. "
            "Please upload some financial documents first, and then I'll be able to help answer your questions.\n\n"
            "📁 You can upload PDFs, text files, or other supported document formats using the upload feature."
        )
        
        return RAGResponse(
            answer=answer,
            sources=[],
            query=question,
            confidence="LOW",
            model=self.llm.model_name,
            num_sources=0,
        )
    
    def _create_no_context_response(self, question: str) -> RAGResponse:
        """Create response when no relevant context is found."""
        answer = (
            f"I couldn't find relevant information in the uploaded documents to answer: \"{question}\"\n\n"
            "**Suggestions:**\n"
            "- Try rephrasing your question\n"
            "- Check if the relevant document has been uploaded\n"
            "- Ask about a specific document or topic that you know is covered\n\n"
            "📋 If you need information not in these documents, please upload additional relevant materials."
        )
        
        return RAGResponse(
            answer=answer,
            sources=[],
            query=question,
            confidence="LOW",
            model=self.llm.model_name,
            num_sources=0,
        )
    
    async def aquery(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> RAGResponse:
        """Async version of query."""
        import time
        start_time = time.time()
        
        # Rewrite query
        rewritten_query = None
        if self.enable_query_rewriting:
            rewritten_query = self._rewrite_query(question)
            search_query = rewritten_query
        else:
            search_query = question
        
        # Retrieve
        try:
            retrieved = self.retriever.retrieve(
                query=search_query,
                top_k=top_k,
                filter=filter,
            )
        except (NoDocumentsError, NoRelevantContextError) as e:
            if isinstance(e, NoDocumentsError):
                return self._create_no_documents_response(question)
            return self._create_no_context_response(question)
        
        # Build context and generate
        context = self._build_context(retrieved)
        sources = self._extract_sources(retrieved)
        prompt = PromptTemplates.format_answer_prompt(context, question)
        
        answer = await self.llm.agenerate(prompt, system_prompt=SYSTEM_PROMPT)
        
        latency_ms = (time.time() - start_time) * 1000
        confidence = self._estimate_confidence(retrieved, answer)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            rewritten_query=rewritten_query,
            confidence=confidence,
            latency_ms=latency_ms,
            model=self.llm.model_name,
            num_sources=len(sources),
        )
