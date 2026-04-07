#!/usr/bin/env python3
"""
RAG Evaluation Script
======================

Evaluate the RAG system using predefined test cases and RAGAS metrics.

Usage:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --output results.json
    python scripts/evaluate_rag.py --test-cases custom_tests.json
    python scripts/evaluate_rag.py --categories factual_extraction comparison

Options:
    --output, -o        Path to save evaluation results (JSON)
    --test-cases, -t    Path to custom test cases JSON file
    --categories, -c    Filter test cases by category
    --verbose, -v       Enable verbose logging
    --skip-ragas        Skip RAGAS metrics (faster)
    --sample, -n        Number of test cases to sample
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.evaluation.test_cases import EVALUATION_TEST_CASES, validate_response
from app.evaluation.metrics import RAGMetrics
from app.evaluation.ragas_eval import RAGASEvaluator
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.generation.chain import RAGChain

logger = get_logger(__name__)


def load_test_cases(
    custom_path: Optional[Path] = None,
    categories: Optional[list[str]] = None,
    sample_size: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Load and filter test cases.
    
    Args:
        custom_path: Path to custom test cases JSON
        categories: Categories to filter by
        sample_size: Number of cases to sample
    
    Returns:
        List of test cases
    """
    
    # Load test cases
    if custom_path and custom_path.exists():
        logger.info(f"Loading custom test cases from {custom_path}")
        with open(custom_path) as f:
            test_cases = json.load(f)
    else:
        test_cases = EVALUATION_TEST_CASES
    
    # Filter by category
    if categories:
        test_cases = [
            tc for tc in test_cases
            if tc.get("category") in categories
        ]
        logger.info(f"Filtered to {len(test_cases)} cases in categories: {categories}")
    
    # Sample
    if sample_size and sample_size < len(test_cases):
        import random
        test_cases = random.sample(test_cases, sample_size)
        logger.info(f"Sampled {sample_size} test cases")
    
    return test_cases


def run_evaluation(
    test_cases: list[dict[str, Any]],
    skip_ragas: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run evaluation on test cases.
    
    Args:
        test_cases: List of test cases to evaluate
        skip_ragas: Skip RAGAS metrics
        verbose: Verbose output
    
    Returns:
        Evaluation results dictionary
    """
    
    # Initialize components
    logger.info("Initializing RAG system...")
    
    embeddings = EmbeddingModel()
    vector_store = VectorStore(embeddings=embeddings)
    retriever = Retriever(vector_store=vector_store)
    rag_chain = RAGChain(retriever=retriever)
    
    # Check if we have documents
    stats = vector_store.get_stats()
    if stats.get("document_count", 0) == 0:
        logger.error("No documents in vector store! Run ingest_documents.py first.")
        return {"error": "No documents in vector store"}
    
    logger.info(f"Vector store has {stats['document_count']} documents")
    
    # Initialize evaluators
    metrics_calculator = RAGMetrics()
    ragas_evaluator = None if skip_ragas else RAGASEvaluator()
    
    # Run evaluation
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_test_cases": len(test_cases),
        "config": {
            "chunk_size": settings.chunk_size,
            "top_k": settings.top_k,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.openai_model,
        },
        "test_results": [],
        "aggregate_metrics": {},
    }
    
    total_latency = 0
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        category = test_case.get("category", "unknown")
        
        logger.info(f"[{i}/{len(test_cases)}] Evaluating: {question[:50]}...")
        
        try:
            # Query RAG system
            start_time = time.time()
            response = rag_chain.query(
                question=question,
                top_k=settings.top_k,
            )
            latency_ms = (time.time() - start_time) * 1000
            total_latency += latency_ms
            
            answer = response.get("answer", "")
            source_docs = response.get("source_documents", [])
            contexts = [doc.page_content for doc in source_docs]
            
            # Validate response
            validation = validate_response(test_case, answer, contexts)
            
            if validation["passed"]:
                passed += 1
                status = "✓ PASS"
            else:
                failed += 1
                status = "✗ FAIL"
            
            # Calculate custom metrics
            custom_metrics = metrics_calculator.evaluate(
                question=question,
                answer=answer,
                contexts=contexts,
            )
            
            # RAGAS metrics (if enabled)
            ragas_metrics = {}
            if ragas_evaluator:
                try:
                    ragas_metrics = ragas_evaluator.evaluate_single(
                        question=question,
                        answer=answer,
                        contexts=contexts,
                    )
                except Exception as e:
                    logger.warning(f"RAGAS evaluation failed: {e}")
            
            # Store result
            test_result = {
                "question": question,
                "category": category,
                "expected_keywords": test_case.get("expected_keywords", []),
                "answer": answer,
                "latency_ms": latency_ms,
                "num_sources": len(source_docs),
                "validation": validation,
                "custom_metrics": custom_metrics,
                "ragas_metrics": ragas_metrics,
                "confidence": response.get("confidence", "MEDIUM"),
            }
            results["test_results"].append(test_result)
            
            if verbose:
                logger.info(f"  {status} | {latency_ms:.0f}ms | {len(source_docs)} sources")
                logger.info(f"  Answer preview: {answer[:100]}...")
            else:
                logger.info(f"  {status} | {latency_ms:.0f}ms")
            
        except Exception as e:
            failed += 1
            logger.error(f"  ✗ ERROR: {e}")
            results["test_results"].append({
                "question": question,
                "category": category,
                "error": str(e),
                "validation": {"passed": False, "reason": f"Error: {e}"},
            })
    
    # Calculate aggregate metrics
    results["aggregate_metrics"] = {
        "pass_rate": passed / len(test_cases) if test_cases else 0,
        "passed": passed,
        "failed": failed,
        "avg_latency_ms": total_latency / len(test_cases) if test_cases else 0,
    }
    
    # Aggregate custom metrics
    all_custom_metrics = [
        r["custom_metrics"] for r in results["test_results"]
        if "custom_metrics" in r
    ]
    
    if all_custom_metrics:
        metric_names = all_custom_metrics[0].keys()
        for name in metric_names:
            values = [m[name] for m in all_custom_metrics if name in m]
            if values:
                results["aggregate_metrics"][f"avg_{name}"] = sum(values) / len(values)
    
    # Aggregate RAGAS metrics
    if not skip_ragas:
        all_ragas_metrics = [
            r["ragas_metrics"] for r in results["test_results"]
            if "ragas_metrics" in r and r["ragas_metrics"]
        ]
        
        if all_ragas_metrics:
            ragas_names = all_ragas_metrics[0].keys()
            for name in ragas_names:
                values = [m.get(name, 0) for m in all_ragas_metrics]
                if values:
                    results["aggregate_metrics"][f"avg_ragas_{name}"] = sum(values) / len(values)
    
    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print evaluation summary."""
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    agg = results["aggregate_metrics"]
    
    print(f"\n📋 Test Cases: {results['num_test_cases']}")
    print(f"✅ Passed: {agg['passed']}")
    print(f"❌ Failed: {agg['failed']}")
    print(f"📈 Pass Rate: {agg['pass_rate']:.1%}")
    print(f"⏱️  Avg Latency: {agg['avg_latency_ms']:.0f}ms")
    
    # Custom metrics
    print("\n📏 Custom Metrics:")
    for key, value in agg.items():
        if key.startswith("avg_") and not key.startswith("avg_ragas"):
            metric_name = key.replace("avg_", "").replace("_", " ").title()
            print(f"   {metric_name}: {value:.2%}")
    
    # RAGAS metrics
    ragas_metrics = {k: v for k, v in agg.items() if k.startswith("avg_ragas")}
    if ragas_metrics:
        print("\n🔬 RAGAS Metrics:")
        for key, value in ragas_metrics.items():
            metric_name = key.replace("avg_ragas_", "").replace("_", " ").title()
            print(f"   {metric_name}: {value:.2%}")
    
    # By category
    print("\n📁 Results by Category:")
    
    category_results = {}
    for result in results["test_results"]:
        cat = result.get("category", "unknown")
        if cat not in category_results:
            category_results[cat] = {"passed": 0, "total": 0}
        category_results[cat]["total"] += 1
        if result.get("validation", {}).get("passed"):
            category_results[cat]["passed"] += 1
    
    for cat, stats in sorted(category_results.items()):
        rate = stats["passed"] / stats["total"] if stats["total"] else 0
        print(f"   {cat}: {stats['passed']}/{stats['total']} ({rate:.0%})")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full evaluation
    python scripts/evaluate_rag.py

    # Save results to file
    python scripts/evaluate_rag.py --output results.json

    # Filter by category
    python scripts/evaluate_rag.py --categories factual_extraction comparison

    # Quick evaluation (skip RAGAS)
    python scripts/evaluate_rag.py --skip-ragas

    # Sample subset
    python scripts/evaluate_rag.py --sample 5 --verbose
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Path to save evaluation results (JSON)"
    )
    
    parser.add_argument(
        "--test-cases", "-t",
        type=Path,
        help="Path to custom test cases JSON file"
    )
    
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        help="Filter test cases by category"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS metrics (faster)"
    )
    
    parser.add_argument(
        "--sample", "-n",
        type=int,
        help="Number of test cases to sample"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.verbose)
    
    # Load test cases
    test_cases = load_test_cases(
        custom_path=args.test_cases,
        categories=args.categories,
        sample_size=args.sample,
    )
    
    if not test_cases:
        logger.error("No test cases to evaluate!")
        sys.exit(1)
    
    logger.info(f"Evaluating {len(test_cases)} test cases...")
    
    # Run evaluation
    results = run_evaluation(
        test_cases=test_cases,
        skip_ragas=args.skip_ragas,
        verbose=args.verbose,
    )
    
    if "error" in results:
        logger.error(results["error"])
        sys.exit(1)
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")
    
    # Exit code based on pass rate
    pass_rate = results["aggregate_metrics"]["pass_rate"]
    if pass_rate < 0.5:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
