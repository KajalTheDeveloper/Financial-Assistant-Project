"""
FastAPI Routes

REST API endpoints for the Financial Knowledge Assistant.
"""

import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import __version__
from app.api.schemas import (
    CollectionStats,
    DocumentInfo,
    DocumentListResponse,
    ErrorResponse,
    FeedbackRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from app.core.config import settings
from app.core.exceptions import (
    DocumentParsingError,
    FileSizeError,
    NoDocumentsError,
    NoRelevantContextError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.ingestion import DocumentLoader, DocumentChunker, MetadataExtractor
from app.retrieval import get_vector_store
from app.generation import RAGChain

logger = get_logger(__name__)

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="Financial Knowledge Assistant API",
    description="RAG-based API for financial document Q&A",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Dependencies
# =============================================================================

def get_rag_chain() -> RAGChain:
    """Get RAG chain instance."""
    return RAGChain()

def get_document_loader() -> DocumentLoader:
    """Get document loader instance."""
    return DocumentLoader()

# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and component status."""
    components = {}
    
    # Check vector store
    try:
        vs = get_vector_store()
        stats = vs.get_collection_stats()
        components["vector_store"] = f"ok ({stats['document_count']} docs)"
    except Exception as e:
        components["vector_store"] = f"error: {str(e)}"
    
    # Check LLM (basic)
    components["llm"] = "ok" if settings.openai_api_key else "missing API key"
    
    return HealthResponse(
        status="healthy" if all("ok" in v for v in components.values()) else "degraded",
        version=__version__,
        timestamp=datetime.now().isoformat(),
        components=components,
    )

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "name": "Financial Knowledge Assistant API",
        "version": __version__,
        "docs": "/docs",
    }

# =============================================================================
# Query Endpoints
# =============================================================================

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base.
    
    Ask questions about the uploaded financial documents.
    """
    logger.info("Query received", question=request.question[:50])
    
    try:
        chain = get_rag_chain()
        response = chain.query(
            question=request.question,
            top_k=request.top_k,
            filter=request.filter,
            stream=False,
        )
        
        # Convert to API response
        return QueryResponse(
            answer=response.answer,
            sources=[
                SourceInfo(**source) for source in response.sources
            ],
            query=response.query,
            rewritten_query=response.rewritten_query,
            confidence=response.confidence,
            latency_ms=response.latency_ms,
            timestamp=response.timestamp,
            model=response.model,
            num_sources=response.num_sources,
        )
        
    except NoDocumentsError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NoRelevantContextError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream", tags=["Query"])
async def query_documents_stream(request: QueryRequest):
    """
    Query with streaming response.
    
    Returns a streaming response for real-time display.
    """
    try:
        chain = get_rag_chain()
        
        async def generate():
            async for chunk in chain.llm.agenerate_stream(
                request.question,
                system_prompt=None  # Will be added by chain
            ):
                yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
        )
        
    except Exception as e:
        logger.error("Streaming query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Document Endpoints
# =============================================================================

@app.post("/documents/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    document_type: Optional[str] = Query(default=None, description="Document type"),
):
    """
    Upload a document for indexing.
    
    Supports PDF, TXT, CSV, DOCX files.
    """
    logger.info("Document upload", filename=file.filename)
    
    # Validate file size
    content = await file.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    try:
        # Load document
        loader = get_document_loader()
        
        # Create a temporary file-like object
        import io
        file_obj = io.BytesIO(content)
        
        metadata = {}
        if document_type:
            metadata["document_type"] = document_type
        
        documents = loader.load(
            file_path=file.filename,
            file_obj=file_obj,
            metadata=metadata,
        )
        
        # Chunk documents
        chunker = DocumentChunker()
        chunks = chunker.chunk(documents)
        
        # Extract metadata
        extractor = MetadataExtractor()
        chunks = extractor.extract_batch(chunks)
        
        # Add to vector store
        vs = get_vector_store()
        vs.add_documents(chunks)
        
        return {
            "status": "success",
            "filename": file.filename,
            "num_pages": len(documents),
            "num_chunks": len(chunks),
            "message": f"Successfully processed and indexed {file.filename}"
        }
        
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except DocumentParsingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    try:
        vs = get_vector_store()
        sources = vs.list_sources()
        
        documents = [
            DocumentInfo(
                filename=source,
                num_chunks=0,  # Would need to count per source
                file_type=source.split(".")[-1] if "." in source else "unknown",
                ingested_at="",
            )
            for source in sources
        ]
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents),
        )
        
    except Exception as e:
        logger.error("List documents failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """Delete a document from the index."""
    try:
        vs = get_vector_store()
        deleted = vs.delete_by_source(filename)
        
        if deleted == 0:
            raise HTTPException(status_code=404, detail=f"Document not found: {filename}")
        
        return {
            "status": "success",
            "filename": filename,
            "chunks_deleted": deleted,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delete document failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/stats", response_model=CollectionStats, tags=["Documents"])
async def get_collection_stats():
    """Get vector store statistics."""
    try:
        vs = get_vector_store()
        stats = vs.get_collection_stats()
        sources = vs.list_sources()
        
        return CollectionStats(
            collection_name=stats["collection_name"],
            document_count=stats["document_count"],
            unique_sources=len(sources),
            sources=sources,
        )
        
    except Exception as e:
        logger.error("Get stats failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Feedback Endpoints
# =============================================================================

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a query response."""
    logger.info(
        "Feedback received",
        query_id=request.query_id,
        rating=request.rating,
    )
    
    # In production, store this in a database
    return {
        "status": "success",
        "message": "Thank you for your feedback!",
    }

# =============================================================================
# Run with: uvicorn app.api.routes:app --reload
# =============================================================================
