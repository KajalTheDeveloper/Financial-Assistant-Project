"""
Custom Exceptions

Defines application-specific exceptions for better error handling.
"""

from typing import Any, Optional


class FinancialAssistantError(Exception):
    """Base exception for Financial Knowledge Assistant."""
    
    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# =============================================================================
# Document Processing Errors
# =============================================================================

class DocumentProcessingError(FinancialAssistantError):
    """Error during document processing."""
    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    """Unsupported file type for processing."""
    
    def __init__(self, file_type: str, supported_types: list[str]) -> None:
        message = f"Unsupported file type: {file_type}. Supported types: {supported_types}"
        super().__init__(message, {"file_type": file_type, "supported_types": supported_types})


class DocumentParsingError(DocumentProcessingError):
    """Error parsing document content."""
    
    def __init__(self, filename: str, reason: str) -> None:
        message = f"Failed to parse document '{filename}': {reason}"
        super().__init__(message, {"filename": filename, "reason": reason})


class EmptyDocumentError(DocumentProcessingError):
    """Document contains no extractable content."""
    
    def __init__(self, filename: str) -> None:
        message = f"Document '{filename}' contains no extractable text content"
        super().__init__(message, {"filename": filename})


# =============================================================================
# Retrieval Errors
# =============================================================================

class RetrievalError(FinancialAssistantError):
    """Error during document retrieval."""
    pass


class VectorStoreError(RetrievalError):
    """Error with vector store operations."""
    pass


class EmbeddingError(RetrievalError):
    """Error generating embeddings."""
    
    def __init__(self, reason: str) -> None:
        message = f"Failed to generate embeddings: {reason}"
        super().__init__(message, {"reason": reason})


class NoDocumentsError(RetrievalError):
    """No documents found in vector store."""
    
    def __init__(self) -> None:
        message = "No documents found. Please upload documents first."
        super().__init__(message)


class NoRelevantContextError(RetrievalError):
    """No relevant context found for query."""
    
    def __init__(self, query: str, threshold: float) -> None:
        message = f"No relevant documents found for query (threshold: {threshold})"
        super().__init__(message, {"query": query, "threshold": threshold})


# =============================================================================
# Generation Errors
# =============================================================================

class GenerationError(FinancialAssistantError):
    """Error during response generation."""
    pass


class LLMError(GenerationError):
    """Error with LLM API call."""
    
    def __init__(self, reason: str, model: str) -> None:
        message = f"LLM error ({model}): {reason}"
        super().__init__(message, {"reason": reason, "model": model})


class RateLimitError(GenerationError):
    """API rate limit exceeded."""
    
    def __init__(self, retry_after: Optional[int] = None) -> None:
        message = "API rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, {"retry_after": retry_after})


class ContextTooLongError(GenerationError):
    """Context exceeds model's token limit."""
    
    def __init__(self, context_tokens: int, max_tokens: int) -> None:
        message = f"Context too long ({context_tokens} tokens). Maximum: {max_tokens}"
        super().__init__(message, {"context_tokens": context_tokens, "max_tokens": max_tokens})


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(FinancialAssistantError):
    """Error in application configuration."""
    pass


class MissingAPIKeyError(ConfigurationError):
    """Required API key is missing."""
    
    def __init__(self, key_name: str) -> None:
        message = f"Missing required API key: {key_name}. Please set it in .env file."
        super().__init__(message, {"key_name": key_name})


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(FinancialAssistantError):
    """Input validation error."""
    pass


class QueryValidationError(ValidationError):
    """Invalid query input."""
    
    def __init__(self, reason: str) -> None:
        message = f"Invalid query: {reason}"
        super().__init__(message, {"reason": reason})


class FileSizeError(ValidationError):
    """File exceeds size limit."""
    
    def __init__(self, file_size: int, max_size: int, filename: str) -> None:
        message = f"File '{filename}' ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({max_size / 1024 / 1024:.1f}MB)"
        super().__init__(message, {
            "filename": filename,
            "file_size": file_size,
            "max_size": max_size
        })


# =============================================================================
# Aliases for backward compatibility
# =============================================================================

RAGError = FinancialAssistantError
