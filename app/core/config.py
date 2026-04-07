"""
Application Configuration

Centralized configuration management using Pydantic Settings.
Loads settings from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==========================================================================
    # API Keys
    # ==========================================================================
    openai_api_key: str = Field(default="", description="OpenAI API key")
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key")
    huggingface_api_key: Optional[str] = Field(default=None, description="HuggingFace API key")
    
    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for generation"
    )
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Embedding model for vectorization"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Cross-encoder model for reranking"
    )
    
    # LLM Parameters
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_max_tokens: int = Field(default=1024, ge=100, le=4096)
    
    # ==========================================================================
    # RAG Configuration
    # ==========================================================================
    # Chunking
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    
    # Retrieval
    top_k_retrieval: int = Field(default=5, ge=1, le=20)
    initial_retrieval_k: int = Field(default=20, ge=5, le=100)
    similarity_threshold: float = Field(default=0.5, ge=0, le=1)
    
    # Feature flags
    enable_reranking: bool = Field(default=True)
    enable_hybrid_search: bool = Field(default=True)
    enable_query_rewriting: bool = Field(default=True)
    
    # ==========================================================================
    # Vector Database
    # ==========================================================================
    vector_db_type: Literal["chroma", "faiss"] = Field(default="chroma")
    chroma_persist_dir: str = Field(default="./data/chroma_db")
    collection_name: str = Field(default="financial_documents")
    
    # ==========================================================================
    # Application Settings
    # ==========================================================================
    app_name: str = Field(default="Financial Knowledge Assistant")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    
    # ==========================================================================
    # API Server
    # ==========================================================================
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    
    # ==========================================================================
    # UI Settings
    # ==========================================================================
    streamlit_port: int = Field(default=8501)
    max_file_size_mb: int = Field(default=50)
    
    # ==========================================================================
    # Evaluation
    # ==========================================================================
    enable_evaluation_logging: bool = Field(default=True)
    eval_sample_size: int = Field(default=100)
    
    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.project_root / "data"
    
    @property
    def raw_data_dir(self) -> Path:
        """Get the raw data directory."""
        return self.data_dir / "raw"
    
    @property
    def sample_docs_dir(self) -> Path:
        """Get the sample documents directory."""
        return self.data_dir / "sample_docs"
    
    @property
    def chroma_path(self) -> Path:
        """Get ChromaDB persistence path."""
        return Path(self.chroma_persist_dir)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env == "production"
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    def validate_api_keys(self) -> bool:
        """Validate that required API keys are set."""
        if not self.openai_api_key:
            return False
        return True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience alias
settings = get_settings()
