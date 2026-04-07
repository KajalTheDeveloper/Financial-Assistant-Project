"""
Metadata Extractor

Extract structured metadata from financial documents.
"""

import re
from datetime import datetime
from typing import Any, Optional

try:
    from langchain_core.documents import Document
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict

from app.core.logging import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    """
    Extract and enrich metadata from financial documents.
    
    Extracts:
    - Document type (annual report, factsheet, circular, etc.)
    - Company/fund names
    - Dates and time periods
    - Financial entities
    """
    
    # Document type patterns
    DOCUMENT_TYPE_PATTERNS = {
        "annual_report": [
            r"annual\s+report",
            r"form\s+10-k",
            r"yearly\s+report",
        ],
        "quarterly_report": [
            r"quarterly\s+report",
            r"form\s+10-q",
            r"q[1-4]\s+\d{4}",
        ],
        "factsheet": [
            r"fact\s*sheet",
            r"fund\s+factsheet",
            r"scheme\s+information",
        ],
        "circular": [
            r"circular",
            r"notification",
            r"sebi.*circular",
            r"rbi.*circular",
        ],
        "prospectus": [
            r"prospectus",
            r"scheme\s+information\s+document",
            r"sid",
        ],
        "policy": [
            r"policy",
            r"guidelines",
            r"regulations",
        ],
    }
    
    # Date patterns
    DATE_PATTERNS = [
        r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",  # DD-MM-YYYY or MM/DD/YYYY
        r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",    # YYYY-MM-DD
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        r"\bFY\s*(\d{4}(?:-\d{2,4})?)\b",  # FY2023-24
        r"\b(Q[1-4]\s+\d{4})\b",  # Q1 2024
    ]
    
    # Financial entity patterns
    ENTITY_PATTERNS = {
        "company": [
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|Inc|Corp|Corporation|Bank|Finance)\.?)\b",
        ],
        "fund": [
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Fund|Scheme|ETF|Index))\b",
        ],
        "regulator": [
            r"\b(SEBI|RBI|SEC|FINRA|AMFI|BSE|NSE)\b",
        ],
    }
    
    def __init__(self):
        # Compile patterns for efficiency
        self.doc_type_compiled = {
            doc_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for doc_type, patterns in self.DOCUMENT_TYPE_PATTERNS.items()
        }
        self.date_compiled = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
        self.entity_compiled = {
            entity_type: [re.compile(p) for p in patterns]
            for entity_type, patterns in self.ENTITY_PATTERNS.items()
        }
    
    def extract(self, document: Document) -> Document:
        """
        Extract and add metadata to a document.
        
        Args:
            document: Document to process
        
        Returns:
            Document with enriched metadata
        """
        content = document.page_content
        metadata = document.metadata.copy()
        
        # Extract document type
        doc_type = self._extract_document_type(content)
        if doc_type:
            metadata["document_type"] = doc_type
        
        # Extract dates
        dates = self._extract_dates(content)
        if dates:
            metadata["extracted_dates"] = dates
            # Try to identify the primary date
            metadata["document_date"] = dates[0] if dates else None
        
        # Extract year
        year = self._extract_year(content, dates)
        if year:
            metadata["document_year"] = year
        
        # Extract entities
        entities = self._extract_entities(content)
        if entities:
            metadata["extracted_entities"] = entities
        
        # Extract company/fund name
        company = self._extract_primary_entity(content, entities)
        if company:
            metadata["company_name"] = company
        
        # Calculate content statistics
        metadata["char_count"] = len(content)
        metadata["word_count"] = len(content.split())
        
        return Document(
            page_content=document.page_content,
            metadata=metadata
        )
    
    def extract_batch(self, documents: list[Document]) -> list[Document]:
        """
        Extract metadata from multiple documents.
        
        Args:
            documents: List of documents to process
        
        Returns:
            Documents with enriched metadata
        """
        return [self.extract(doc) for doc in documents]
    
    def _extract_document_type(self, content: str) -> Optional[str]:
        """Identify document type from content."""
        # Check first 2000 characters (usually contains document type)
        sample = content[:2000].lower()
        
        for doc_type, patterns in self.doc_type_compiled.items():
            for pattern in patterns:
                if pattern.search(sample):
                    return doc_type
        
        return "general"
    
    def _extract_dates(self, content: str) -> list[str]:
        """Extract dates from content."""
        dates = []
        
        # Search in first portion of document
        sample = content[:5000]
        
        for pattern in self.date_compiled:
            matches = pattern.findall(sample)
            dates.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_dates = []
        for date in dates:
            if date.lower() not in seen:
                seen.add(date.lower())
                unique_dates.append(date)
        
        return unique_dates[:5]  # Return top 5 dates
    
    def _extract_year(self, content: str, dates: list[str]) -> Optional[int]:
        """Extract the primary year from dates or content."""
        current_year = datetime.now().year
        
        # Try to find year from dates
        for date_str in dates:
            # Look for 4-digit year
            match = re.search(r"(\d{4})", date_str)
            if match:
                year = int(match.group(1))
                if 1990 <= year <= current_year + 1:
                    return year
        
        # Search content for fiscal year
        fy_match = re.search(r"FY\s*(\d{4})", content[:3000], re.IGNORECASE)
        if fy_match:
            return int(fy_match.group(1))
        
        return None
    
    def _extract_entities(self, content: str) -> dict[str, list[str]]:
        """Extract named entities from content."""
        entities: dict[str, list[str]] = {}
        sample = content[:5000]
        
        for entity_type, patterns in self.entity_compiled.items():
            found = []
            for pattern in patterns:
                matches = pattern.findall(sample)
                found.extend(matches)
            
            # Deduplicate
            if found:
                entities[entity_type] = list(set(found))[:5]
        
        return entities
    
    def _extract_primary_entity(
        self,
        content: str,
        entities: dict[str, list[str]]
    ) -> Optional[str]:
        """Identify the primary company or fund name."""
        # Check if we found any companies or funds
        companies = entities.get("company", [])
        funds = entities.get("fund", [])
        
        # Prefer fund names for factsheets, company names for reports
        sample = content[:1000].lower()
        
        if "fund" in sample or "scheme" in sample:
            if funds:
                return funds[0]
        
        if companies:
            return companies[0]
        
        if funds:
            return funds[0]
        
        return None


def enrich_metadata(documents: list[Document]) -> list[Document]:
    """
    Convenience function to enrich document metadata.
    
    Args:
        documents: Documents to process
    
    Returns:
        Documents with enriched metadata
    """
    extractor = MetadataExtractor()
    return extractor.extract_batch(documents)
