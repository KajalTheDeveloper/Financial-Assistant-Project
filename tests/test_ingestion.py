"""
Tests for the Ingestion Module
"""

import pytest
from pathlib import Path

# Import actual module classes
from app.ingestion.document_loader import DocumentLoader, SUPPORTED_EXTENSIONS
from app.ingestion.chunker import DocumentChunker
from app.ingestion.preprocessor import TextPreprocessor
from app.ingestion.metadata_extractor import MetadataExtractor

# Import Document type consistently with the module
try:
    from langchain_core.documents import Document
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict


class TestDocumentLoader:
    """Tests for DocumentLoader class."""

    def test_init(self):
        loader = DocumentLoader()
        assert loader is not None

    def test_supported_extensions(self):
        loader = DocumentLoader()
        supported = loader.supported_extensions
        assert ".pdf" in supported
        assert ".txt" in supported
        assert ".csv" in supported

    def test_load_txt_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_content = "This is a test document.\nIt has multiple lines."
        test_file.write_text(test_content)

        loader = DocumentLoader()
        documents = loader.load(test_file)

        assert len(documents) >= 1
        assert "test document" in documents[0].page_content.lower()

    def test_load_nonexistent_file(self):
        loader = DocumentLoader()
        with pytest.raises(Exception):
            loader.load(Path("/nonexistent/file.pdf"))

    def test_load_unsupported_extension(self, tmp_path):
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")

        loader = DocumentLoader()
        with pytest.raises(Exception):
            loader.load(test_file)


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    def test_init_default_params(self):
        chunker = DocumentChunker()
        # Should not raise and have default values
        assert chunker.chunk_size > 0
        assert chunker.chunk_overlap >= 0

    def test_init_custom_params(self):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_chunk_short_text(self):
        chunker = DocumentChunker(chunk_size=1000)
        docs = [Document(page_content="Short text", metadata={"source": "test"})]
        chunks = chunker.chunk(docs)
        assert len(chunks) >= 1
        assert "Short text" in chunks[0].page_content

    def test_chunk_long_text(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        long_text = "This is a sentence. " * 50
        docs = [Document(page_content=long_text, metadata={"source": "test"})]
        chunks = chunker.chunk(docs)
        assert len(chunks) >= 1

    def test_metadata_preserved(self):
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        docs = [Document(page_content="A" * 200, metadata={"source": "test.pdf", "page": 1})]
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            assert chunk.metadata.get("source_file") or chunk.metadata.get("source")


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_init(self):
        preprocessor = TextPreprocessor()
        assert preprocessor is not None

    def test_normalize_whitespace(self):
        preprocessor = TextPreprocessor()
        text = "Multiple   spaces   and\n\n\nnewlines"
        result = preprocessor.preprocess(text)
        assert "   " not in result or len(result) < len(text)

    def test_preserve_content(self):
        preprocessor = TextPreprocessor()
        text = "Revenue increased by 15% in Q3 2023."
        result = preprocessor.preprocess(text)
        assert "Revenue" in result or "revenue" in result
        assert "15" in result

    def test_empty_string(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("")
        assert result == ""


class TestMetadataExtractor:
    """Tests for MetadataExtractor class."""

    def test_init(self):
        extractor = MetadataExtractor()
        assert extractor is not None

    def test_extract_from_document(self):
        extractor = MetadataExtractor()
        doc = Document(page_content="Apple Inc. reported revenue for 2023.", metadata={})
        result = extractor.extract(doc)
        # extract() returns a Document with enriched metadata
        assert hasattr(result, 'metadata')
        assert isinstance(result.metadata, dict)

    def test_extract_from_document_with_dates(self):
        extractor = MetadataExtractor()
        doc = Document(page_content="The report covers fiscal year 2023 ending December 31, 2023.", metadata={})
        result = extractor.extract(doc)
        # extract() returns a Document with enriched metadata
        assert hasattr(result, 'metadata')
        assert isinstance(result.metadata, dict)


# Fixtures
@pytest.fixture
def sample_documents():
    return [
        Document(page_content="Apple Inc. reported Q3 2023 revenue of $81.8 billion.", metadata={"source": "apple_10q.pdf", "page": 1}),
        Document(page_content="Net income was $19.9 billion with EPS of $1.26.", metadata={"source": "apple_10q.pdf", "page": 2}),
    ]


@pytest.fixture
def chunker():
    return DocumentChunker(chunk_size=500, chunk_overlap=50)


@pytest.fixture
def preprocessor():
    return TextPreprocessor()
