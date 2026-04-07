"""
Metrics Dashboard Component

Display evaluation metrics and analytics.
"""

import streamlit as st
from typing import Any, Optional
from datetime import datetime


def render_metrics_dashboard(
    metrics: dict[str, Any],
    show_details: bool = True,
) -> None:
    """
    Render the evaluation metrics dashboard.
    
    Args:
        metrics: Dictionary of metric name -> value
        show_details: Whether to show detailed breakdown
    """
    
    st.markdown("## 📊 RAG Evaluation Metrics")
    
    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        faithfulness = metrics.get("faithfulness", 0)
        st.metric(
            "Faithfulness",
            f"{faithfulness:.0%}",
            help="How well answers are grounded in context"
        )
        render_metric_bar(faithfulness)
    
    with col2:
        relevancy = metrics.get("answer_relevancy", 0)
        st.metric(
            "Answer Relevancy",
            f"{relevancy:.0%}",
            help="How well answers address the question"
        )
        render_metric_bar(relevancy)
    
    with col3:
        precision = metrics.get("context_precision", 0)
        st.metric(
            "Context Precision",
            f"{precision:.0%}",
            help="Relevance of retrieved documents"
        )
        render_metric_bar(precision)
    
    with col4:
        overall = metrics.get("overall_score", 0)
        st.metric(
            "Overall Score",
            f"{overall:.0%}",
            help="Weighted average of all metrics"
        )
        render_metric_bar(overall)
    
    if show_details:
        st.divider()
        render_detailed_metrics(metrics)


def render_metric_bar(value: float, color: str = None) -> None:
    """Render a small progress bar for a metric."""
    
    if color is None:
        if value >= 0.8:
            color = "#28a745"  # Green
        elif value >= 0.6:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
    
    st.markdown(
        f"""<div style="
            background: #e0e0e0;
            border-radius: 4px;
            height: 8px;
            margin-top: -10px;
        ">
            <div style="
                background: {color};
                width: {value * 100}%;
                height: 100%;
                border-radius: 4px;
            "></div>
        </div>""",
        unsafe_allow_html=True
    )


def render_detailed_metrics(metrics: dict[str, Any]) -> None:
    """Render detailed metric breakdown."""
    
    with st.expander("📈 Detailed Metrics", expanded=False):
        # Custom metrics
        if "custom_metrics" in metrics:
            st.markdown("### Custom Metrics")
            
            custom = metrics["custom_metrics"]
            for name, value in custom.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(name.replace("_", " ").title())
                with col2:
                    st.write(f"{value:.2%}")
        
        # Retrieval stats
        st.markdown("### Retrieval Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Avg. Retrieval Time",
                f"{metrics.get('avg_retrieval_ms', 0):.0f}ms"
            )
            st.metric(
                "Avg. Documents Retrieved",
                f"{metrics.get('avg_docs_retrieved', 0):.1f}"
            )
        
        with col2:
            st.metric(
                "Avg. Generation Time",
                f"{metrics.get('avg_generation_ms', 0):.0f}ms"
            )
            st.metric(
                "Cache Hit Rate",
                f"{metrics.get('cache_hit_rate', 0):.0%}"
            )


def render_query_analytics(
    queries: list[dict[str, Any]],
    title: str = "Recent Queries"
) -> None:
    """Render analytics for recent queries."""
    
    st.markdown(f"### 📝 {title}")
    
    if not queries:
        st.info("No queries recorded yet")
        return
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    
    total_queries = len(queries)
    avg_latency = sum(q.get("latency_ms", 0) for q in queries) / total_queries
    avg_sources = sum(q.get("num_sources", 0) for q in queries) / total_queries
    
    with col1:
        st.metric("Total Queries", total_queries)
    with col2:
        st.metric("Avg. Latency", f"{avg_latency:.0f}ms")
    with col3:
        st.metric("Avg. Sources", f"{avg_sources:.1f}")
    
    # Query table
    st.markdown("#### Query History")
    
    for i, query in enumerate(reversed(queries[-10:])):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                q_text = query.get("question", "")
                st.write(q_text[:50] + "..." if len(q_text) > 50 else q_text)
            
            with col2:
                st.write(f"{query.get('latency_ms', 0):.0f}ms")
            
            with col3:
                confidence = query.get("confidence", "MEDIUM")
                emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")
                st.write(emoji)
            
            with col4:
                st.write(f"{query.get('num_sources', 0)} docs")


def render_feedback_summary(
    feedback: list[dict[str, Any]]
) -> None:
    """Render summary of user feedback."""
    
    st.markdown("### 👍 User Feedback")
    
    if not feedback:
        st.info("No feedback collected yet")
        return
    
    # Calculate stats
    total = len(feedback)
    positive = sum(1 for f in feedback if f.get("rating", 0) >= 4)
    negative = sum(1 for f in feedback if f.get("rating", 0) <= 2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Feedback", total)
    with col2:
        st.metric("Positive", f"{positive} ({positive/total*100:.0f}%)")
    with col3:
        st.metric("Negative", f"{negative} ({negative/total*100:.0f}%)")
    
    # Rating distribution
    st.markdown("#### Rating Distribution")
    
    ratings = [0, 0, 0, 0, 0]  # 1-5 stars
    for f in feedback:
        rating = f.get("rating", 0)
        if 1 <= rating <= 5:
            ratings[rating - 1] += 1
    
    for i, count in enumerate(ratings, 1):
        st.progress(count / total if total > 0 else 0, text=f"{'⭐' * i}: {count}")


def render_performance_chart(
    data: list[dict[str, Any]],
    metric: str = "latency_ms",
    title: str = "Performance Over Time"
) -> None:
    """Render a simple performance trend chart."""
    
    st.markdown(f"### 📈 {title}")
    
    if not data:
        st.info("Not enough data for chart")
        return
    
    # Extract values
    values = [d.get(metric, 0) for d in data]
    
    # Simple line chart using Streamlit
    st.line_chart(values)
