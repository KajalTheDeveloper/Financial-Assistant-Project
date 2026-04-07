"""
RAGAS Evaluation

Integration with RAGAS framework for standardized RAG evaluation.
"""

from typing import Any, Optional
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RAGASResult:
    """RAGAS evaluation results."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    details: dict[str, Any]


class RAGASEvaluator:
    """
    RAGAS-based evaluation for RAG systems.
    
    Metrics:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are the retrieved contexts relevant?
    - Context Recall: Are all relevant contexts retrieved?
    """
    
    def __init__(self, llm_model: Optional[str] = None):
        """
        Initialize RAGAS evaluator.
        
        Args:
            llm_model: LLM model for evaluation (uses OpenAI by default)
        """
        self.llm_model = llm_model
        self._ragas_available = self._check_ragas()
    
    def _check_ragas(self) -> bool:
        """Check if RAGAS is available."""
        try:
            import ragas
            return True
        except ImportError:
            logger.warning("RAGAS not installed. Using fallback metrics.")
            return False
    
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> RAGASResult:
        """
        Evaluate a single RAG response.
        
        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved context documents
            ground_truth: Optional ground truth answer
        
        Returns:
            RAGASResult with scores
        """
        if self._ragas_available:
            return self._evaluate_with_ragas(
                question, answer, contexts, ground_truth
            )
        else:
            return self._evaluate_fallback(
                question, answer, contexts, ground_truth
            )
    
    def _evaluate_with_ragas(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> RAGASResult:
        """Evaluate using RAGAS library."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
            
            # Prepare data
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            
            if ground_truth:
                data["ground_truth"] = [ground_truth]
                metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            else:
                metrics = [faithfulness, answer_relevancy, context_precision]
            
            dataset = Dataset.from_dict(data)
            
            # Run evaluation
            result = evaluate(dataset, metrics=metrics)
            
            return RAGASResult(
                faithfulness=result.get("faithfulness", 0.0),
                answer_relevancy=result.get("answer_relevancy", 0.0),
                context_precision=result.get("context_precision", 0.0),
                context_recall=result.get("context_recall", 0.0) if ground_truth else 0.0,
                overall_score=sum(result.values()) / len(result) if result else 0.0,
                details=dict(result),
            )
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return self._evaluate_fallback(question, answer, contexts, ground_truth)
    
    def _evaluate_fallback(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> RAGASResult:
        """Fallback evaluation using simple metrics."""
        from app.evaluation.metrics import RAGMetrics
        
        context_str = "\n".join(contexts)
        
        # Use our custom metrics
        results = RAGMetrics.evaluate_all(
            question=question,
            answer=answer,
            context=context_str,
            sources=[{"source_file": f"context_{i}"} for i in range(len(contexts))],
        )
        
        return RAGASResult(
            faithfulness=results["context_coverage"].score,
            answer_relevancy=results["answer_relevance"].score,
            context_precision=0.8,  # Placeholder
            context_recall=0.8 if ground_truth else 0.0,  # Placeholder
            overall_score=RAGMetrics.aggregate_scores(results),
            details={
                "custom_metrics": {k: v.score for k, v in results.items()},
                "note": "Using fallback metrics (RAGAS not available)",
            },
        )
    
    def evaluate_batch(
        self,
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of dicts with question, answer, contexts keys
        
        Returns:
            Aggregated results
        """
        results = []
        
        for case in test_cases:
            result = self.evaluate(
                question=case["question"],
                answer=case["answer"],
                contexts=case["contexts"],
                ground_truth=case.get("ground_truth"),
            )
            results.append(result)
        
        # Aggregate
        if not results:
            return {"error": "No results"}
        
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        avg_precision = sum(r.context_precision for r in results) / len(results)
        avg_recall = sum(r.context_recall for r in results) / len(results)
        avg_overall = sum(r.overall_score for r in results) / len(results)
        
        return {
            "num_cases": len(results),
            "avg_faithfulness": avg_faithfulness,
            "avg_answer_relevancy": avg_relevancy,
            "avg_context_precision": avg_precision,
            "avg_context_recall": avg_recall,
            "avg_overall_score": avg_overall,
            "individual_results": [r.details for r in results],
        }
