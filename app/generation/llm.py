"""
LLM Generator

Interface for LLM providers with streaming support.
"""

from functools import lru_cache
from typing import AsyncIterator, Iterator, Optional

from langchain_openai import ChatOpenAI
# LangChain v1->v2 compatibility: use langchain_core document/message types
try:
    # langchain_core provides Document and message wrappers
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
except Exception:
    # Fallback to simple local message dataclasses
    from dataclasses import dataclass

    @dataclass
    class SystemMessage:
        content: str

    @dataclass
    class HumanMessage:
        content: str

    @dataclass
    class AIMessage:
        content: str

from app.core.config import settings
from app.core.exceptions import LLMError, MissingAPIKeyError, RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMGenerator:
    """
    LLM interface with support for multiple providers.
    
    Features:
    - Streaming responses
    - Retry logic
    - Token counting
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        streaming: bool = True,
    ):
        """
        Initialize LLM generator.
        
        Args:
            model_name: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            streaming: Enable streaming
        """
        self.model_name = model_name or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.streaming = streaming
        
        # Validate API key
        if not settings.openai_api_key:
            raise MissingAPIKeyError("OPENAI_API_KEY")
        
        logger.info(
            "Initializing LLM",
            model=self.model_name,
            temperature=self.temperature
        )
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=self.streaming,
            api_key=settings.openai_api_key,
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response (non-streaming).
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
        
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError()
            raise LLMError(str(e), self.model_name)
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Generate a streaming response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
        
        Yields:
            Response chunks
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        try:
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError()
            raise LLMError(str(e), self.model_name)
    
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Async generate a response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
        
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError()
            raise LLMError(str(e), self.model_name)
    
    async def agenerate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Async generate a streaming response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
        
        Yields:
            Response chunks
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        try:
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError()
            raise LLMError(str(e), self.model_name)
    
    def get_langchain_llm(self) -> ChatOpenAI:
        """Get the underlying LangChain LLM object."""
        return self.llm


@lru_cache(maxsize=1)
def get_llm(
    model_name: Optional[str] = None,
    streaming: bool = True,
) -> LLMGenerator:
    """
    Get cached LLM generator.
    
    Args:
        model_name: Optional model override
        streaming: Enable streaming
    
    Returns:
        LLMGenerator instance
    """
    return LLMGenerator(model_name=model_name, streaming=streaming)
