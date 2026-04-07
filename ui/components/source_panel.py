"""
Source Panel Component

Display source citations and document previews.
"""

import streamlit as st
from typing import Any, Optional


def render_source_panel(
    sources: list[dict[str, Any]],
    expanded: bool = True,
) -> None:
    """
    Render the source citations panel.
    
    Args:
        sources: List of source documents with metadata
        expanded: Whether panel should be expanded by default
    """
    if not sources:
        return
    
    with st.expander(
        f"📚 Source Citations ({len(sources)} documents)",
        expanded=expanded
    ):
        for i, source in enumerate(sources):
            render_source_card(source, index=i)


def render_source_card(
    source: dict[str, Any],
    index: int = 0,
) -> None:
    """Render a single source card."""
    
    filename = source.get("source_file", "Unknown Document")
    page = source.get("page_number", "N/A")
    score = source.get("score", 0)
    preview = source.get("preview", "")
    chunk_type = source.get("chunk_type", "text")
    
    # Card container
    with st.container():
        # Header row
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # File icon based on type
            file_ext = filename.split(".")[-1].lower() if "." in filename else ""
            icon = {
                "pdf": "📕",
                "txt": "📄",
                "csv": "📊",
                "docx": "📘",
                "md": "📝",
            }.get(file_ext, "📄")
            
            st.markdown(f"### {icon} {filename}")
        
        with col2:
            st.markdown(f"**Page:** {page}")
        
        with col3:
            if score > 0:
                # Color code the relevance score
                color = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
                st.markdown(f"**Relevance:** {color} {score:.0%}")
        
        # Chunk type badge
        type_colors = {
            "text": "blue",
            "table": "green",
        }
        st.markdown(
            f"<span style='background-color: #e0e0e0; padding: 2px 8px; "
            f"border-radius: 4px; font-size: 0.8rem;'>{chunk_type}</span>",
            unsafe_allow_html=True
        )
        
        # Preview content
        if preview:
            st.markdown("**Preview:**")
            st.text_area(
                label="Content preview",
                value=preview,
                height=100,
                disabled=True,
                key=f"preview_{index}",
                label_visibility="collapsed"
            )
        
        st.divider()


def render_source_summary(sources: list[dict[str, Any]]) -> None:
    """Render a compact summary of sources."""
    
    if not sources:
        return
    
    # Group by source file
    by_file: dict[str, list] = {}
    for source in sources:
        filename = source.get("source_file", "Unknown")
        if filename not in by_file:
            by_file[filename] = []
        by_file[filename].append(source)
    
    st.markdown("**📚 Sources:**")
    
    for filename, file_sources in by_file.items():
        pages = [str(s.get("page_number", "?")) for s in file_sources]
        avg_score = sum(s.get("score", 0) for s in file_sources) / len(file_sources)
        
        st.markdown(
            f"- **{filename}** (Pages: {', '.join(pages)}) "
            f"- Avg relevance: {avg_score:.0%}"
        )


def render_source_comparison(
    sources_list: list[list[dict[str, Any]]],
    labels: list[str],
) -> None:
    """
    Compare sources from multiple queries side by side.
    
    Useful for showing how different queries retrieved different documents.
    """
    
    if len(sources_list) != len(labels):
        st.error("Number of source lists must match number of labels")
        return
    
    cols = st.columns(len(sources_list))
    
    for i, (sources, label) in enumerate(zip(sources_list, labels)):
        with cols[i]:
            st.markdown(f"### {label}")
            
            if not sources:
                st.info("No sources found")
                continue
            
            for source in sources[:5]:  # Show max 5
                filename = source.get("source_file", "Unknown")
                page = source.get("page_number", "N/A")
                score = source.get("score", 0)
                
                st.markdown(f"""
                <div style="
                    background: #f8f9fa;
                    padding: 8px;
                    border-radius: 4px;
                    margin-bottom: 8px;
                ">
                    <strong>{filename}</strong><br/>
                    <span style="font-size: 0.8rem; color: #666;">
                        Page {page} | {score:.0%} match
                    </span>
                </div>
                """, unsafe_allow_html=True)


def render_document_viewer(
    content: str,
    filename: str,
    page: int = 1,
) -> None:
    """
    Render a document content viewer.
    
    For displaying full document content or specific pages.
    """
    
    st.markdown(f"### 📄 {filename} - Page {page}")
    
    # Toolbar
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("⬅️ Previous", disabled=page <= 1):
            pass  # Handle navigation
    
    with col2:
        if st.button("➡️ Next"):
            pass  # Handle navigation
    
    with col3:
        search_term = st.text_input(
            "Search in document",
            placeholder="Find...",
            key=f"search_{filename}",
            label_visibility="collapsed"
        )
    
    # Content area
    display_content = content
    
    if search_term:
        # Highlight search term
        import re
        display_content = re.sub(
            f"({re.escape(search_term)})",
            r"**\1**",
            content,
            flags=re.IGNORECASE
        )
    
    st.markdown(
        f"""<div style="
            background: white;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Georgia', serif;
            line-height: 1.6;
        ">{display_content}</div>""",
        unsafe_allow_html=True
    )
