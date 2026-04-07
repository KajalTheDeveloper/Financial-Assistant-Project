"""
Response Formatter

Format RAG responses with proper citations and structure.
"""

import re
from typing import Any, Optional


class ResponseFormatter:
    """
    Format responses for display.
    
    Features:
    - Citation formatting
    - Confidence badges
    - Source cards
    - Markdown formatting
    """
    
    def __init__(self):
        self.confidence_badges = {
            "HIGH": "🟢 HIGH",
            "MEDIUM": "🟡 MEDIUM",
            "LOW": "🔴 LOW",
        }
    
    def format_response(
        self,
        answer: str,
        sources: list[dict[str, Any]],
        confidence: str = "MEDIUM",
        latency_ms: float = 0.0,
        include_disclaimer: bool = True,
    ) -> str:
        """
        Format a complete response for display.
        
        Args:
            answer: The generated answer
            sources: List of source documents
            confidence: Confidence level
            latency_ms: Response latency
            include_disclaimer: Whether to add disclaimer
        
        Returns:
            Formatted markdown string
        """
        parts = []
        
        # Main answer
        parts.append(answer)
        
        # Sources section
        if sources:
            parts.append("\n\n---\n")
            parts.append("**📚 Sources:**\n")
            for source in sources:
                parts.append(self._format_source(source))
        
        # Metadata footer
        parts.append("\n\n---\n")
        parts.append(f"*Confidence: {self.confidence_badges.get(confidence, confidence)}*")
        if latency_ms > 0:
            parts.append(f" | *Response time: {latency_ms:.0f}ms*")
        
        # Disclaimer
        if include_disclaimer:
            parts.append("\n\n")
            parts.append(
                "📋 *This information is for educational purposes only. "
                "Please consult a qualified financial advisor for personalized advice.*"
            )
        
        return "".join(parts)
    
    def _format_source(self, source: dict[str, Any]) -> str:
        """Format a single source citation."""
        index = source.get("index", "?")
        filename = source.get("source_file", "Unknown")
        page = source.get("page_number", "N/A")
        score = source.get("score", 0)
        
        score_display = f" (relevance: {score:.0%})" if score > 0 else ""
        
        return f"- [{index}] **{filename}**, Page {page}{score_display}\n"
    
    def format_source_card(self, source: dict[str, Any]) -> dict[str, Any]:
        """
        Format a source for card display in UI.
        
        Returns a dict suitable for Streamlit cards.
        """
        return {
            "title": source.get("source_file", "Unknown Document"),
            "page": f"Page {source.get('page_number', 'N/A')}",
            "score": source.get("score", 0),
            "score_display": f"{source.get('score', 0):.0%}",
            "preview": source.get("preview", ""),
            "chunk_type": source.get("chunk_type", "text"),
        }
    
    def add_citations_to_text(
        self,
        text: str,
        sources: list[dict[str, Any]]
    ) -> str:
        """
        Add inline citations to text if missing.
        
        This is a simple heuristic - in production you might use
        more sophisticated citation matching.
        """
        # Check if text already has citations
        if re.search(r'\[Source:', text):
            return text
        
        # For now, add a general citation note
        if sources:
            source_names = [s.get("source_file", "Unknown") for s in sources[:3]]
            citation = f"\n\n*Based on: {', '.join(source_names)}*"
            return text + citation
        
        return text
    
    def format_comparison_table(
        self,
        items: list[dict[str, Any]],
        metrics: list[str]
    ) -> str:
        """
        Format a comparison as a markdown table.
        
        Args:
            items: List of items to compare
            metrics: List of metric names
        
        Returns:
            Markdown table string
        """
        if not items or not metrics:
            return ""
        
        # Header
        headers = ["Metric"] + [item.get("name", f"Item {i+1}") for i, item in enumerate(items)]
        header_row = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        # Data rows
        rows = [header_row, separator]
        for metric in metrics:
            row_data = [metric]
            for item in items:
                value = item.get(metric, "N/A")
                row_data.append(str(value))
            rows.append("| " + " | ".join(row_data) + " |")
        
        return "\n".join(rows)
    
    def format_risk_list(self, risks: list[dict[str, Any]]) -> str:
        """Format a list of risks for display."""
        if not risks:
            return "No specific risks identified in the documents."
        
        parts = []
        for risk in risks:
            category = risk.get("category", "General")
            description = risk.get("description", "")
            severity = risk.get("severity", "Unknown")
            source = risk.get("source", "")
            
            severity_icon = {
                "High": "🔴",
                "Medium": "🟡",
                "Low": "🟢",
            }.get(severity, "⚪")
            
            parts.append(
                f"- **{category}** {severity_icon}\n"
                f"  {description}\n"
                f"  *Source: {source}*\n"
            )
        
        return "\n".join(parts)
    
    def truncate_for_preview(
        self,
        text: str,
        max_length: int = 200
    ) -> str:
        """Truncate text for preview with ellipsis."""
        if len(text) <= max_length:
            return text
        
        # Try to cut at a word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        
        if last_space > max_length * 0.7:
            truncated = truncated[:last_space]
        
        return truncated + "..."
