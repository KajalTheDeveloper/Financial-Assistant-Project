"""
Text Preprocessor

Clean and normalize text before embedding.
"""

import re
import unicodedata
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Text preprocessing for financial documents.
    
    Operations:
    - Unicode normalization
    - Whitespace normalization
    - Special character handling
    - Financial number formatting
    """
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_extra_whitespace: bool = True,
        normalize_unicode: bool = True,
        fix_encoding: bool = True,
        normalize_numbers: bool = False,
    ):
        """
        Initialize preprocessor.
        
        Args:
            lowercase: Convert to lowercase
            remove_extra_whitespace: Normalize whitespace
            normalize_unicode: Normalize Unicode characters
            fix_encoding: Fix common encoding issues
            normalize_numbers: Standardize number formats
        """
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.fix_encoding = fix_encoding
        self.normalize_numbers = normalize_numbers
    
    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.
        
        Args:
            text: Raw text to preprocess
        
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Unicode normalization
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Fix encoding issues
        if self.fix_encoding:
            text = self._fix_encoding(text)
        
        # Normalize whitespace
        if self.remove_extra_whitespace:
            text = self._normalize_whitespace(text)
        
        # Normalize numbers
        if self.normalize_numbers:
            text = self._normalize_numbers(text)
        
        # Lowercase (usually not recommended for embeddings)
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def preprocess_batch(self, texts: list[str]) -> list[str]:
        """Preprocess multiple texts."""
        return [self.preprocess(text) for text in texts]
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # NFKC normalization: compatibility decomposition followed by canonical composition
        text = unicodedata.normalize("NFKC", text)
        
        # Replace common problematic characters
        replacements = {
            "\u2018": "'",   # Left single quote
            "\u2019": "'",   # Right single quote
            "\u201c": '"',   # Left double quote
            "\u201d": '"',   # Right double quote
            "\u2013": "-",   # En dash
            "\u2014": "-",   # Em dash
            "\u2026": "...", # Ellipsis
            "\u00a0": " ",   # Non-breaking space
            "\u00ad": "",    # Soft hyphen
            "\ufeff": "",    # BOM
            "\u200b": "",    # Zero-width space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Fix double-encoded UTF-8
        try:
            # Check if text looks double-encoded
            if "â€" in text or "Ã" in text:
                # Try to fix
                text = text.encode("latin-1").decode("utf-8", errors="ignore")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        
        # Remove null bytes
        text = text.replace("\x00", "")
        
        # Remove other control characters except newlines and tabs
        text = "".join(
            char for char in text
            if char in "\n\t" or not unicodedata.category(char).startswith("C")
        )
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Replace tabs with spaces
        text = text.replace("\t", "    ")
        
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split("\n")]
        
        # Remove multiple blank lines (keep max 2)
        result_lines = []
        blank_count = 0
        
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)
        
        text = "\n".join(result_lines)
        
        # Normalize multiple spaces within lines (but preserve indentation)
        lines = text.split("\n")
        normalized_lines = []
        
        for line in lines:
            # Preserve leading whitespace
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            
            # Normalize spaces in content
            normalized = " ".join(stripped.split())
            
            normalized_lines.append(indent + normalized)
        
        return "\n".join(normalized_lines)
    
    def _normalize_numbers(self, text: str) -> str:
        """Standardize number formats for better matching."""
        # Normalize currency symbols
        currency_map = {
            "₹": "INR ",
            "$": "USD ",
            "€": "EUR ",
            "£": "GBP ",
        }
        
        for symbol, replacement in currency_map.items():
            text = text.replace(symbol, replacement)
        
        # Normalize percentage formats
        text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", text)
        
        # Normalize large numbers with suffixes
        # e.g., "10L" -> "10 Lakh", "5Cr" -> "5 Crore"
        text = re.sub(r"(\d+(?:\.\d+)?)\s*[Ll]akh?", r"\1 Lakh", text)
        text = re.sub(r"(\d+(?:\.\d+)?)\s*[Cc]r(?:ore)?", r"\1 Crore", text)
        text = re.sub(r"(\d+(?:\.\d+)?)\s*[Mm](?:illion)?", r"\1 Million", text)
        text = re.sub(r"(\d+(?:\.\d+)?)\s*[Bb](?:illion)?", r"\1 Billion", text)
        
        return text
    
    def clean_for_display(self, text: str) -> str:
        """
        Clean text for display in UI.
        
        Args:
            text: Text to clean
        
        Returns:
            Display-ready text
        """
        # Basic normalization
        text = self._normalize_unicode(text)
        text = self._normalize_whitespace(text)
        
        # Truncate very long lines
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            if len(line) > 500:
                line = line[:500] + "..."
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)


def clean_text(text: str, **kwargs) -> str:
    """
    Convenience function for text cleaning.
    
    Args:
        text: Text to clean
        **kwargs: Preprocessor options
    
    Returns:
        Cleaned text
    """
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.preprocess(text)
