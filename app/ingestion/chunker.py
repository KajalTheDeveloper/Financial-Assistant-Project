"""
Document Chunker

Smart chunking strategies for financial documents.
Preserves context and respects document structure.
"""

from typing import Optional

try:
    from langchain_core.documents import Document
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Lightweight fallback splitter used for tests and portability.
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, length_function=len, separators=None, keep_separator=True):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function
            self.separators = separators or ["\n\n", "\n", ". ", " "]
            self.keep_separator = keep_separator

        def split_text(self, text: str) -> list[str]:
            if not text:
                return []

            n = len(text)
            if n <= self.chunk_size:
                return [text]

            chunks = []
            start = 0
            while start < n:
                end = min(start + self.chunk_size, n)
                chunk = text[start:end]
                chunks.append(chunk)
                # Move with overlap
                start = max(end - self.chunk_overlap, end)

            return chunks

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """
    Intelligent document chunking with multiple strategies.
    
    Supports:
    - Recursive character splitting (default)
    - Section-aware splitting for structured documents
    - Table-preserving chunking
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        length_function: callable = len,
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            length_function: Function to measure text length
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.length_function = length_function
        
        # Financial document separators (in order of priority)
        self.separators = [
            "\n\n\n",      # Major section breaks
            "\n\n",        # Paragraph breaks
            "\n",          # Line breaks
            ". ",          # Sentences
            ", ",          # Clauses
            " ",           # Words
            "",            # Characters (last resort)
        ]
        
        # Initialize the base splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            separators=self.separators,
            keep_separator=True,
        )
    
    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into chunks while preserving metadata.
        
        Args:
            documents: List of Document objects to chunk
        
        Returns:
            List of chunked Document objects with updated metadata
        """
        all_chunks = []
        
        for doc in documents:
            # Preprocess content
            content = self._preprocess_content(doc.page_content)
            
            # Check if document contains tables
            has_tables = "[TABLES]" in content
            
            if has_tables:
                chunks = self._chunk_with_tables(content, doc.metadata)
            else:
                chunks = self._chunk_standard(content, doc.metadata)
            
            all_chunks.extend(chunks)
        
        # Add chunk indices
        source_chunk_counts = {}
        for chunk in all_chunks:
            source = chunk.metadata.get("source_file", "unknown")
            if source not in source_chunk_counts:
                source_chunk_counts[source] = 0
            chunk.metadata["chunk_index"] = source_chunk_counts[source]
            source_chunk_counts[source] += 1
        
        # Add total chunks count
        for chunk in all_chunks:
            source = chunk.metadata.get("source_file", "unknown")
            chunk.metadata["total_chunks"] = source_chunk_counts[source]
        
        logger.info(
            "Chunking complete",
            total_chunks=len(all_chunks),
            documents=len(documents)
        )
        
        return all_chunks
    
    def _preprocess_content(self, content: str) -> str:
        """Clean and normalize content before chunking."""
        # Normalize whitespace
        lines = content.split("\n")
        cleaned_lines = []
        
        for line in lines:
            # Remove excessive whitespace within lines
            cleaned = " ".join(line.split())
            cleaned_lines.append(cleaned)
        
        # Rejoin with single newlines, then normalize paragraph breaks
        content = "\n".join(cleaned_lines)
        
        # Normalize multiple newlines to double newlines (paragraph break)
        while "\n\n\n" in content:
            content = content.replace("\n\n\n", "\n\n")
        
        return content.strip()
    
    def _chunk_standard(
        self,
        content: str,
        metadata: dict
    ) -> list[Document]:
        """Standard recursive chunking."""
        texts = self.splitter.split_text(content)
        
        chunks = []
        for i, text in enumerate(texts):
            if text.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_type"] = "text"
                chunks.append(Document(
                    page_content=text,
                    metadata=chunk_metadata
                ))
        
        return chunks
    
    def _chunk_with_tables(
        self,
        content: str,
        metadata: dict
    ) -> list[Document]:
        """Chunk content while preserving table integrity."""
        chunks = []
        
        # Split by table marker
        parts = content.split("[TABLES]")
        
        # Process text before tables
        if parts[0].strip():
            text_chunks = self.splitter.split_text(parts[0])
            for text in text_chunks:
                if text.strip():
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_type"] = "text"
                    chunks.append(Document(
                        page_content=text,
                        metadata=chunk_metadata
                    ))
        
        # Process tables (keep as single chunks if possible)
        if len(parts) > 1:
            table_content = parts[1]
            
            # If table is small enough, keep it whole
            if len(table_content) <= self.chunk_size * 1.5:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_type"] = "table"
                chunks.append(Document(
                    page_content=f"[TABLE DATA]\n{table_content}",
                    metadata=chunk_metadata
                ))
            else:
                # Split large tables
                table_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size * 2,  # Larger chunks for tables
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", " | ", " "],
                )
                table_chunks = table_splitter.split_text(table_content)
                for table_text in table_chunks:
                    if table_text.strip():
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_type"] = "table"
                        chunks.append(Document(
                            page_content=f"[TABLE DATA]\n{table_text}",
                            metadata=chunk_metadata
                        ))
        
        return chunks
    
    def chunk_by_section(
        self,
        documents: list[Document],
        section_markers: Optional[list[str]] = None
    ) -> list[Document]:
        """
        Chunk documents by section headers.
        
        Useful for structured financial documents with clear sections.
        
        Args:
            documents: Documents to chunk
            section_markers: List of section header patterns
        
        Returns:
            Chunked documents
        """
        if section_markers is None:
            # Common financial document section markers
            section_markers = [
                "RISK FACTORS",
                "MANAGEMENT'S DISCUSSION",
                "FINANCIAL STATEMENTS",
                "NOTES TO FINANCIAL",
                "EXECUTIVE SUMMARY",
                "INVESTMENT OBJECTIVE",
                "KEY INFORMATION",
            ]
        
        all_chunks = []
        
        for doc in documents:
            content = doc.page_content
            current_section = "Introduction"
            section_content = ""
            
            lines = content.split("\n")
            
            for line in lines:
                # Check if line is a section header
                is_header = any(
                    marker.lower() in line.lower()
                    for marker in section_markers
                )
                
                if is_header and section_content.strip():
                    # Save previous section
                    section_chunks = self._chunk_standard(
                        section_content,
                        {**doc.metadata, "section": current_section}
                    )
                    all_chunks.extend(section_chunks)
                    
                    # Start new section
                    current_section = line.strip()[:50]  # Truncate long headers
                    section_content = line + "\n"
                else:
                    section_content += line + "\n"
            
            # Don't forget the last section
            if section_content.strip():
                section_chunks = self._chunk_standard(
                    section_content,
                    {**doc.metadata, "section": current_section}
                )
                all_chunks.extend(section_chunks)
        
        return all_chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """
    Convenience function to chunk documents.
    
    Args:
        documents: Documents to chunk
        chunk_size: Optional custom chunk size
        chunk_overlap: Optional custom overlap
    
    Returns:
        List of chunked documents
    """
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return chunker.chunk(documents)
