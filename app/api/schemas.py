"""
API Schemas

Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    filter: Optional[dict[str, Any]] = Field(default=None, description="Metadata filter")
    stream: bool = Field(default=False, description="Enable streaming response")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the key risks in HDFC Balanced Advantage Fund?",
                "top_k": 5,
                "stream": False
            }
        }
    }


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    filename: str = Field(..., description="Name of the file")
    content_type: str = Field(..., description="MIME type of the file")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional metadata")


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    query_id: str = Field(..., description="ID of the query")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_type: str = Field(default="general", description="Type of feedback")
    comment: Optional[str] = Field(default=None, max_length=500, description="Optional comment")


# =============================================================================
# Response Models
# =============================================================================

class SourceInfo(BaseModel):
    """Source document information."""
    index: int
    source_file: str
    page_number: int | str
    score: float
    chunk_type: str
    preview: str


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str
    sources: list[SourceInfo]
    query: str
    rewritten_query: Optional[str] = None
    confidence: str
    latency_ms: float
    timestamp: str
    model: str
    num_sources: int
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "The key risks include market risk, credit risk, and liquidity risk...",
                "sources": [
                    {
                        "index": 1,
                        "source_file": "hdfc_factsheet.pdf",
                        "page_number": 5,
                        "score": 0.89,
                        "chunk_type": "text",
                        "preview": "Risk Factors: The fund is subject to..."
                    }
                ],
                "query": "What are the key risks?",
                "confidence": "HIGH",
                "latency_ms": 1234.5,
                "timestamp": "2024-01-15T10:30:00",
                "model": "gpt-4o-mini",
                "num_sources": 3
            }
        }
    }


class DocumentInfo(BaseModel):
    """Information about a processed document."""
    filename: str
    num_chunks: int
    file_type: str
    ingested_at: str
    metadata: Optional[dict[str, Any]] = None


class DocumentListResponse(BaseModel):
    """Response with list of documents."""
    documents: list[DocumentInfo]
    total_count: int


class CollectionStats(BaseModel):
    """Vector store collection statistics."""
    collection_name: str
    document_count: int
    unique_sources: int
    sources: list[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    components: dict[str, str]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
