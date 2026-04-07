#!/usr/bin/env python3
"""
Document Ingestion Script
==========================

Batch ingest documents into the vector store.

Usage:
    python scripts/ingest_documents.py --input ./data/documents
    python scripts/ingest_documents.py --input ./data/documents --clear
    python scripts/ingest_documents.py --input report.pdf --single

Options:
    --input, -i     Path to documents directory or single file
    --clear, -c     Clear existing collection before ingesting
    --single, -s    Input is a single file, not a directory
    --recursive, -r Include subdirectories
    --verbose, -v   Enable verbose logging
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.ingestion.preprocessor import TextPreprocessor
from app.ingestion.metadata_extractor import MetadataExtractor
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


def ingest_documents(
    input_path: Path,
    clear_existing: bool = False,
    single_file: bool = False,
    recursive: bool = True,
    verbose: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Ingest documents into the vector store.
    
    Args:
        input_path: Path to documents or single file
        clear_existing: Clear collection before ingesting
        single_file: Input is a single file
        recursive: Include subdirectories
        verbose: Enable verbose logging
    
    Returns:
        Tuple of (files_processed, chunks_created, errors)
    """
    
    # Initialize components
    logger.info("Initializing ingestion pipeline...")
    
    loader = DocumentLoader()
    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    preprocessor = TextPreprocessor()
    metadata_extractor = MetadataExtractor()
    embeddings = EmbeddingModel()
    vector_store = VectorStore(embeddings=embeddings)
    
    # Clear if requested
    if clear_existing:
        logger.warning("Clearing existing collection...")
        vector_store.clear()
    
    # Collect files to process
    if single_file:
        if not input_path.is_file():
            logger.error(f"File not found: {input_path}")
            return 0, 0, [f"File not found: {input_path}"]
        files = [input_path]
    else:
        if not input_path.is_dir():
            logger.error(f"Directory not found: {input_path}")
            return 0, 0, [f"Directory not found: {input_path}"]
        
        # Supported extensions
        extensions = {"*.pdf", "*.txt", "*.csv", "*.docx", "*.md", "*.html"}
        
        files = []
        for ext in extensions:
            if recursive:
                files.extend(input_path.rglob(ext))
            else:
                files.extend(input_path.glob(ext))
        
        files = sorted(set(files))
    
    if not files:
        logger.warning("No files found to process")
        return 0, 0, []
    
    logger.info(f"Found {len(files)} file(s) to process")
    
    # Process files
    files_processed = 0
    total_chunks = 0
    errors = []
    
    for i, file_path in enumerate(files, 1):
        try:
            logger.info(f"[{i}/{len(files)}] Processing: {file_path.name}")
            
            # Load document
            documents = loader.load(file_path)
            
            if not documents:
                logger.warning(f"No content extracted from {file_path.name}")
                continue
            
            # Preprocess
            for doc in documents:
                doc.page_content = preprocessor.preprocess(doc.page_content)
                
                # Extract and merge metadata
                extracted_meta = metadata_extractor.extract(doc.page_content)
                doc.metadata.update(extracted_meta)
            
            # Chunk
            chunks = chunker.chunk_documents(documents)
            
            if verbose:
                logger.debug(f"  Created {len(chunks)} chunks")
                for j, chunk in enumerate(chunks[:3]):
                    logger.debug(f"  Chunk {j+1}: {len(chunk.page_content)} chars")
            
            # Add to vector store
            vector_store.add_documents(chunks)
            
            files_processed += 1
            total_chunks += len(chunks)
            
            logger.info(f"  ✓ Added {len(chunks)} chunks")
            
        except Exception as e:
            error_msg = f"{file_path.name}: {str(e)}"
            errors.append(error_msg)
            logger.error(f"  ✗ Failed: {e}")
    
    # Summary
    logger.info("=" * 50)
    logger.info("Ingestion Complete!")
    logger.info(f"  Files processed: {files_processed}/{len(files)}")
    logger.info(f"  Total chunks: {total_chunks}")
    if errors:
        logger.info(f"  Errors: {len(errors)}")
    
    # Get final stats
    stats = vector_store.get_stats()
    logger.info(f"  Collection size: {stats.get('document_count', 0)} documents")
    
    return files_processed, total_chunks, errors


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Ingest documents into the vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ingest all documents in a directory
    python scripts/ingest_documents.py --input ./data/documents

    # Clear and reingest
    python scripts/ingest_documents.py --input ./data/documents --clear

    # Ingest a single file
    python scripts/ingest_documents.py --input report.pdf --single

    # Verbose mode
    python scripts/ingest_documents.py --input ./data/documents --verbose
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to documents directory or single file"
    )
    
    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear existing collection before ingesting"
    )
    
    parser.add_argument(
        "--single", "-s",
        action="store_true",
        help="Input is a single file, not a directory"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Include subdirectories (default: True)"
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not include subdirectories"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.verbose)
    
    # Run ingestion
    files_processed, chunks_created, errors = ingest_documents(
        input_path=args.input,
        clear_existing=args.clear,
        single_file=args.single,
        recursive=args.recursive,
        verbose=args.verbose,
    )
    
    # Exit code
    if errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
