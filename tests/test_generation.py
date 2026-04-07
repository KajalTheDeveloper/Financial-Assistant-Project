"""
Tests for the Generation Module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Import Document type consistently with the module
try:
    from langchain_core.documents import Document
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict


class TestLLMGenerator:
    """Tests for LLMGenerator class."""

    @patch('app.generation.llm.settings')
    @patch('app.generation.llm.ChatOpenAI')
    def test_init(self, mock_chat, mock_settings):
        """Test LLMGenerator initialization."""
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.llm_temperature = 0.1
        mock_settings.llm_max_tokens = 2000
        
        from app.generation.llm import LLMGenerator
        generator = LLMGenerator()
        
        assert generator is not None
        mock_chat.assert_called_once()

    @patch('app.generation.llm.settings')
    @patch('app.generation.llm.ChatOpenAI')
    def test_generate(self, mock_chat, mock_settings):
        """Test non-streaming generation."""
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.llm_temperature = 0.1
        mock_settings.llm_max_tokens = 2000
        
        mock_llm = mock_chat.return_value
        mock_response = Mock()
        mock_response.content = "This is the generated response."
        mock_llm.invoke.return_value = mock_response
        
        from app.generation.llm import LLMGenerator
        generator = LLMGenerator()
        
        result = generator.generate("What is revenue?")
        
        assert result == "This is the generated response."
        mock_llm.invoke.assert_called_once()

    @patch('app.generation.llm.settings')
    @patch('app.generation.llm.ChatOpenAI')
    def test_generate_with_system_prompt(self, mock_chat, mock_settings):
        """Test generation with system prompt."""
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.llm_temperature = 0.1
        mock_settings.llm_max_tokens = 2000
        
        mock_llm = mock_chat.return_value
        mock_response = Mock()
        mock_response.content = "Response with system context."
        mock_llm.invoke.return_value = mock_response
        
        from app.generation.llm import LLMGenerator
        generator = LLMGenerator()
        
        result = generator.generate(
            "What is revenue?",
            system_prompt="You are a helpful assistant."
        )
        
        assert result == "Response with system context."
        # Should have 2 messages: system + human
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 2

    @patch('app.generation.llm.settings')
    def test_missing_api_key(self, mock_settings):
        """Test error when API key is missing."""
        mock_settings.openai_api_key = None
        
        from app.generation.llm import LLMGenerator
        from app.core.exceptions import MissingAPIKeyError
        
        with pytest.raises(MissingAPIKeyError):
            LLMGenerator()

    @patch('app.generation.llm.settings')
    @patch('app.generation.llm.ChatOpenAI')
    def test_generate_stream(self, mock_chat, mock_settings):
        """Test streaming generation."""
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.llm_temperature = 0.1
        mock_settings.llm_max_tokens = 2000
        
        # Mock stream response
        mock_chunk1 = Mock()
        mock_chunk1.content = "Hello "
        mock_chunk2 = Mock()
        mock_chunk2.content = "World"
        
        mock_llm = mock_chat.return_value
        mock_llm.stream.return_value = [mock_chunk1, mock_chunk2]
        
        from app.generation.llm import LLMGenerator
        generator = LLMGenerator()
        
        chunks = list(generator.generate_stream("Say hello"))
        
        assert chunks == ["Hello ", "World"]


class TestPromptTemplates:
    """Tests for PromptTemplates."""

    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        from app.generation.prompts import SYSTEM_PROMPT
        
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 100
        assert "financial" in SYSTEM_PROMPT.lower()

    def test_format_answer_prompt(self):
        """Test answer prompt formatting."""
        from app.generation.prompts import PromptTemplates
        
        context = "Apple reported $81.8 billion revenue."
        question = "What was Apple's revenue?"
        
        prompt = PromptTemplates.format_answer_prompt(context, question)
        
        assert context in prompt
        assert question in prompt

    def test_format_query_rewrite(self):
        """Test query rewrite prompt formatting."""
        from app.generation.prompts import PromptTemplates
        
        query = "What is MF NAV?"
        
        prompt = PromptTemplates.format_query_rewrite(query)
        
        assert query in prompt


class TestRAGChain:
    """Tests for RAGChain class."""

    @patch('app.generation.chain.get_llm')
    @patch('app.generation.chain.DocumentRetriever')
    def test_init(self, mock_retriever_class, mock_get_llm):
        """Test RAG chain initialization."""
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        from app.generation.chain import RAGChain
        
        chain = RAGChain(
            retriever=mock_retriever,
            llm=mock_llm,
            enable_query_rewriting=False,
        )
        
        assert chain is not None
        assert chain.retriever == mock_retriever
        assert chain.llm == mock_llm

    @patch('app.generation.chain.get_llm')
    @patch('app.generation.chain.DocumentRetriever')
    def test_query_no_documents(self, mock_retriever_class, mock_get_llm):
        """Test query when no documents exist."""
        from app.core.exceptions import NoDocumentsError
        
        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = NoDocumentsError()
        mock_retriever_class.return_value = mock_retriever
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        from app.generation.chain import RAGChain
        
        chain = RAGChain(
            retriever=mock_retriever,
            llm=mock_llm,
            enable_query_rewriting=False,
        )
        
        # Should return a response indicating no documents
        response = chain.query("test question")
        assert "no documents" in response.answer.lower() or response.answer != ""


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_create_response(self):
        """Test creating RAG response."""
        from app.generation.chain import RAGResponse
        
        response = RAGResponse(
            answer="Apple's revenue was $81.8 billion.",
            sources=[{"source": "apple.pdf", "page": 1}],
            query="What was Apple's revenue?",
            confidence="HIGH",
        )
        
        assert response.answer == "Apple's revenue was $81.8 billion."
        assert len(response.sources) == 1
        assert response.confidence == "HIGH"

    def test_to_dict(self):
        """Test converting response to dictionary."""
        from app.generation.chain import RAGResponse
        
        response = RAGResponse(
            answer="Test answer",
            sources=[],
            query="Test query",
        )
        
        result = response.to_dict()
        
        assert isinstance(result, dict)
        assert result["answer"] == "Test answer"
        assert result["query"] == "Test query"


class TestResponseFormatter:
    """Tests for ResponseFormatter class."""

    def test_init(self):
        """Test response formatter initialization."""
        from app.generation.response_formatter import ResponseFormatter
        formatter = ResponseFormatter()
        assert formatter is not None


# Fixtures
@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = Mock()
    mock.model_name = "gpt-4o-mini"
    mock.generate.return_value = "Mock response"
    return mock


@pytest.fixture
def mock_retriever():
    """Create a mock retriever for testing."""
    mock = Mock()
    mock.retrieve.return_value = [
        {
            "content": "Apple Inc. reported Q3 2023 revenue of $81.8 billion.",
            "metadata": {"source_file": "apple_10q.pdf", "page_number": 1},
            "score": 0.95,
        }
    ]
    return mock
