"""
Tests for the Retrieval Module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import Document type consistently with the module
try:
    from langchain_core.documents import Document
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    @patch('app.retrieval.embeddings.HuggingFaceEmbeddings')
    def test_init(self, mock_hf):
        """Test initialization with default parameters."""
        mock_hf.return_value.embed_query.return_value = [0.1] * 384
        
        from app.retrieval.embeddings import EmbeddingGenerator
        generator = EmbeddingGenerator()
        
        assert generator is not None
        assert generator.dimension > 0

    @patch('app.retrieval.embeddings.HuggingFaceEmbeddings')
    def test_embed_query(self, mock_hf):
        """Test single query embedding."""
        expected_embedding = [0.1] * 384
        mock_hf.return_value.embed_query.return_value = expected_embedding
        
        from app.retrieval.embeddings import EmbeddingGenerator
        generator = EmbeddingGenerator()
        
        result = generator.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 384

    @patch('app.retrieval.embeddings.HuggingFaceEmbeddings')
    def test_embed_documents(self, mock_hf):
        """Test document embedding."""
        expected_embeddings = [[0.1] * 384, [0.2] * 384]
        mock_hf.return_value.embed_query.return_value = [0.1] * 384
        mock_hf.return_value.embed_documents.return_value = expected_embeddings
        
        from app.retrieval.embeddings import EmbeddingGenerator
        generator = EmbeddingGenerator()
        
        result = generator.embed_documents(["doc 1", "doc 2"])
        assert len(result) == 2

    @patch('app.retrieval.embeddings.HuggingFaceEmbeddings')
    def test_embed_empty_list(self, mock_hf):
        """Test embedding empty document list."""
        mock_hf.return_value.embed_query.return_value = [0.1] * 384
        
        from app.retrieval.embeddings import EmbeddingGenerator
        generator = EmbeddingGenerator()
        
        result = generator.embed_documents([])
        assert result == []


class TestReranker:
    """Tests for Reranker class."""

    @patch('app.retrieval.reranker.CrossEncoder')
    def test_init(self, mock_cross_encoder):
        """Test reranker initialization."""
        from app.retrieval.reranker import Reranker
        reranker = Reranker()
        assert reranker is not None

    @patch('app.retrieval.reranker.CrossEncoder')
    def test_rerank(self, mock_cross_encoder):
        """Test document reranking."""
        mock_instance = mock_cross_encoder.return_value
        mock_instance.predict.return_value = np.array([0.9, 0.3, 0.7])
        
        from app.retrieval.reranker import Reranker
        reranker = Reranker()
        
        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
            Document(page_content="Doc 3", metadata={}),
        ]
        
        result = reranker.rerank("query", docs, top_k=2)
        assert len(result) == 2
        assert result[0].page_content == "Doc 1"

    @patch('app.retrieval.reranker.CrossEncoder')
    def test_rerank_empty_docs(self, mock_cross_encoder):
        """Test reranking empty document list."""
        from app.retrieval.reranker import Reranker
        reranker = Reranker()
        
        result = reranker.rerank("query", [])
        assert result == []

    @patch('app.retrieval.reranker.CrossEncoder')
    def test_rerank_with_scores(self, mock_cross_encoder):
        """Test reranking with scores returned."""
        mock_instance = mock_cross_encoder.return_value
        mock_instance.predict.return_value = np.array([0.8, 0.5])
        
        from app.retrieval.reranker import Reranker
        reranker = Reranker()
        
        docs = [
            Document(page_content="Doc A", metadata={}),
            Document(page_content="Doc B", metadata={}),
        ]
        
        result = reranker.rerank_with_scores("query", docs)
        assert len(result) == 2
        assert isinstance(result[0], tuple)


class TestHybridSearcher:
    """Tests for HybridSearcher class."""

    def test_tokenize(self):
        """Test tokenization method."""
        from app.retrieval.hybrid_search import HybridSearcher
        
        mock_vs = Mock()
        mock_vs.client.get_collection.return_value.get.return_value = {
            "documents": [],
            "metadatas": []
        }
        
        searcher = HybridSearcher(mock_vs)
        
        tokens = searcher._tokenize("Hello World Test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_init_with_empty_store(self):
        """Test initialization with empty vector store."""
        from app.retrieval.hybrid_search import HybridSearcher
        
        mock_vs = Mock()
        mock_vs.client.get_collection.return_value.get.return_value = {
            "documents": [],
            "metadatas": []
        }
        
        searcher = HybridSearcher(mock_vs)
        assert searcher.bm25 is None


class TestVectorStore:
    """Tests for VectorStore class."""

    @patch('app.retrieval.vector_store.chromadb.PersistentClient')
    @patch('app.retrieval.vector_store.Chroma')
    @patch('app.retrieval.vector_store.get_embeddings')
    def test_init(self, mock_get_emb, mock_chroma, mock_client, tmp_path):
        """Test vector store initialization."""
        mock_emb = Mock()
        mock_emb.get_langchain_embeddings.return_value = Mock()
        mock_get_emb.return_value = mock_emb
        
        from app.retrieval.vector_store import VectorStore
        
        store = VectorStore(
            collection_name="test",
            persist_directory=str(tmp_path / "chroma")
        )
        
        assert store is not None
        assert store.collection_name == "test"

    @patch('app.retrieval.vector_store.chromadb.PersistentClient')
    @patch('app.retrieval.vector_store.Chroma')
    @patch('app.retrieval.vector_store.get_embeddings')
    def test_add_documents(self, mock_get_emb, mock_chroma, mock_client, tmp_path):
        """Test adding documents to store."""
        mock_emb = Mock()
        mock_emb.get_langchain_embeddings.return_value = Mock()
        mock_get_emb.return_value = mock_emb
        
        mock_vectorstore = Mock()
        mock_vectorstore.add_documents.return_value = ["id1", "id2"]
        mock_chroma.return_value = mock_vectorstore
        
        from app.retrieval.vector_store import VectorStore
        
        store = VectorStore(
            collection_name="test",
            persist_directory=str(tmp_path / "chroma")
        )
        
        docs = [
            Document(page_content="Test doc 1", metadata={"source": "test1"}),
            Document(page_content="Test doc 2", metadata={"source": "test2"}),
        ]
        
        result = store.add_documents(docs)
        assert len(result) >= 0


class TestDocumentRetriever:
    """Tests for DocumentRetriever class."""

    @patch('app.retrieval.retriever.get_vector_store')
    @patch('app.retrieval.retriever.Reranker')
    def test_init(self, mock_reranker, mock_get_vs):
        """Test retriever initialization."""
        mock_vs = Mock()
        mock_get_vs.return_value = mock_vs
        
        from app.retrieval.retriever import DocumentRetriever
        
        retriever = DocumentRetriever(
            vector_store=mock_vs,
            enable_reranking=False,
            enable_hybrid_search=False,
        )
        
        assert retriever is not None
        assert retriever.top_k > 0

    @patch('app.retrieval.retriever.get_vector_store')
    def test_retrieve_empty_store(self, mock_get_vs):
        """Test retrieval from empty store raises error."""
        mock_vs = Mock()
        mock_vs.is_empty.return_value = True
        mock_get_vs.return_value = mock_vs
        
        from app.retrieval.retriever import DocumentRetriever
        from app.core.exceptions import NoDocumentsError
        
        retriever = DocumentRetriever(
            vector_store=mock_vs,
            enable_reranking=False,
            enable_hybrid_search=False,
        )
        
        with pytest.raises(NoDocumentsError):
            retriever.retrieve("test query")


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="Apple Inc. reported Q3 2023 revenue.", metadata={"source": "apple.pdf"}),
        Document(page_content="Microsoft announced new cloud services.", metadata={"source": "msft.pdf"}),
    ]
