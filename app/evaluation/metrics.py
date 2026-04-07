"""
RAG Evaluation Metrics

Custom metrics for evaluating RAG quality.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import re


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"{self.metric_name}: {self.score:.2%}"


class RAGMetrics:
    """
    Custom metrics for RAG evaluation.
    
    Metrics:
    - Citation accuracy
    - Context coverage
    - Answer relevance (simple)
    - Hallucination indicators
    """
    
    @staticmethod
    def citation_accuracy(
        answer: str,
        sources: list[dict[str, Any]],
    ) -> EvaluationResult:
        """
        Check if citations in answer match provided sources.
        
        Returns:
            Score between 0 and 1
        """
        # Find citations in answer
        citation_pattern = r'\[Source:\s*([^\]]+)\]|\[(\d+)\]'
        found_citations = re.findall(citation_pattern, answer)
        
        if not found_citations:
            # No citations found - check if sources were provided
            if sources:
                return EvaluationResult(
                    metric_name="citation_accuracy",
                    score=0.0,
                    details={"reason": "No citations found but sources provided"}
                )
            return EvaluationResult(
                metric_name="citation_accuracy",
                score=1.0,
                details={"reason": "No sources to cite"}
            )
        
        # Get source filenames
        source_names = {s.get("source_file", "").lower() for s in sources}
        source_indices = {str(i) for i in range(1, len(sources) + 1)}
        
        # Check each citation
        valid_count = 0
        for citation in found_citations:
            # citation is a tuple from the regex groups
            citation_text = citation[0] or citation[1]
            citation_text = citation_text.lower().strip()
            
            # Check if it matches a source name or index
            if any(name in citation_text for name in source_names):
                valid_count += 1
            elif citation_text in source_indices:
                valid_count += 1
        
        score = valid_count / len(found_citations) if found_citations else 1.0
        
        return EvaluationResult(
            metric_name="citation_accuracy",
            score=score,
            details={
                "total_citations": len(found_citations),
                "valid_citations": valid_count,
            }
        )
    
    @staticmethod
    def context_coverage(
        answer: str,
        context: str,
        min_overlap_ratio: float = 0.3,
    ) -> EvaluationResult:
        """
        Check how much of the answer is supported by context.
        
        Uses simple word overlap as a proxy.
        """
        # Tokenize (simple)
        def get_content_words(text: str) -> set[str]:
            words = re.findall(r'\b\w+\b', text.lower())
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                'through', 'during', 'before', 'after', 'above', 'below',
                'between', 'under', 'again', 'further', 'then', 'once',
                'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                'very', 'just', 'also', 'now', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            }
            return {w for w in words if w not in stop_words and len(w) > 2}
        
        answer_words = get_content_words(answer)
        context_words = get_content_words(context)
        
        if not answer_words:
            return EvaluationResult(
                metric_name="context_coverage",
                score=1.0,
                details={"reason": "Empty answer"}
            )
        
        overlap = answer_words & context_words
        coverage = len(overlap) / len(answer_words)
        
        return EvaluationResult(
            metric_name="context_coverage",
            score=coverage,
            details={
                "answer_words": len(answer_words),
                "context_words": len(context_words),
                "overlap_words": len(overlap),
            }
        )
    
    @staticmethod
    def answer_relevance(
        question: str,
        answer: str,
    ) -> EvaluationResult:
        """
        Simple relevance check based on question word presence.
        
        For production, use embedding similarity or LLM-as-judge.
        """
        def get_question_keywords(text: str) -> set[str]:
            words = re.findall(r'\b\w+\b', text.lower())
            # Remove question words
            question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'is', 'are', 'does', 'do', 'can'}
            return {w for w in words if w not in question_words and len(w) > 3}
        
        question_keywords = get_question_keywords(question)
        answer_lower = answer.lower()
        
        if not question_keywords:
            return EvaluationResult(
                metric_name="answer_relevance",
                score=1.0,
                details={"reason": "No keywords in question"}
            )
        
        found_keywords = sum(1 for kw in question_keywords if kw in answer_lower)
        relevance = found_keywords / len(question_keywords)
        
        return EvaluationResult(
            metric_name="answer_relevance",
            score=relevance,
            details={
                "question_keywords": list(question_keywords),
                "found_in_answer": found_keywords,
            }
        )
    
    @staticmethod
    def hallucination_indicators(
        answer: str,
        context: str,
    ) -> EvaluationResult:
        """
        Check for potential hallucination indicators.
        
        Looks for:
        - Specific numbers not in context
        - Dates not in context
        - Named entities not in context
        """
        # Extract numbers from answer and context
        answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', answer))
        context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', context))
        
        # Extract dates
        date_pattern = r'\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'
        answer_dates = set(re.findall(date_pattern, answer, re.IGNORECASE))
        context_dates = set(re.findall(date_pattern, context, re.IGNORECASE))
        
        # Check for numbers not in context
        unsupported_numbers = answer_numbers - context_numbers
        unsupported_dates = answer_dates - context_dates
        
        # Calculate score (higher = less hallucination risk)
        total_specific = len(answer_numbers) + len(answer_dates)
        unsupported = len(unsupported_numbers) + len(unsupported_dates)
        
        if total_specific == 0:
            score = 1.0
        else:
            score = 1.0 - (unsupported / total_specific)
        
        return EvaluationResult(
            metric_name="hallucination_indicators",
            score=max(0, score),
            details={
                "unsupported_numbers": list(unsupported_numbers),
                "unsupported_dates": list(unsupported_dates),
                "risk_level": "low" if score > 0.8 else "medium" if score > 0.5 else "high",
            }
        )
    
    @staticmethod
    def response_completeness(
        answer: str,
        min_length: int = 50,
        max_length: int = 2000,
    ) -> EvaluationResult:
        """
        Check if response has appropriate length and structure.
        """
        length = len(answer)
        
        # Check length
        if length < min_length:
            length_score = length / min_length
        elif length > max_length:
            length_score = max(0, 1 - (length - max_length) / max_length)
        else:
            length_score = 1.0
        
        # Check structure (has bullet points, paragraphs, etc.)
        has_structure = bool(
            re.search(r'[-•*]\s', answer) or  # Bullet points
            re.search(r'\d+\.\s', answer) or  # Numbered list
            answer.count('\n\n') > 0          # Paragraphs
        )
        
        structure_score = 1.0 if has_structure else 0.7
        
        final_score = (length_score + structure_score) / 2
        
        return EvaluationResult(
            metric_name="response_completeness",
            score=final_score,
            details={
                "length": length,
                "has_structure": has_structure,
                "length_score": length_score,
            }
        )
    
    @classmethod
    def evaluate_all(
        cls,
        question: str,
        answer: str,
        context: str,
        sources: list[dict[str, Any]],
    ) -> dict[str, EvaluationResult]:
        """
        Run all metrics on a single response.
        
        Returns:
            Dictionary of metric name -> result
        """
        return {
            "citation_accuracy": cls.citation_accuracy(answer, sources),
            "context_coverage": cls.context_coverage(answer, context),
            "answer_relevance": cls.answer_relevance(question, answer),
            "hallucination_indicators": cls.hallucination_indicators(answer, context),
            "response_completeness": cls.response_completeness(answer),
        }
    
    @classmethod
    def aggregate_scores(
        cls,
        results: dict[str, EvaluationResult]
    ) -> float:
        """Calculate weighted average of all metrics."""
        weights = {
            "citation_accuracy": 0.2,
            "context_coverage": 0.25,
            "answer_relevance": 0.25,
            "hallucination_indicators": 0.2,
            "response_completeness": 0.1,
        }
        
        total = sum(
            results[name].score * weights.get(name, 0.1)
            for name in results
        )
        
        return total
