"""
Sidebar Component

Document management and settings panel.
"""

import streamlit as st
from pathlib import Path
from typing import Callable, Optional

from app.core.config import settings


def render_sidebar(
    on_file_upload: Callable,
    documents: list[str],
    on_delete_document: Optional[Callable] = None,
    collection_stats: Optional[dict] = None,
) -> dict:
    """
    Render the sidebar with document management and settings.
    
    Args:
        on_file_upload: Callback when files are uploaded
        documents: List of uploaded document names
        on_delete_document: Callback to delete a document
        collection_stats: Vector store statistics
    
    Returns:
        Dictionary of current settings
    """
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 1.5rem; margin: 0;">🏦</h1>
            <h2 style="font-size: 1.1rem; margin: 0.5rem 0;">Financial Knowledge</h2>
            <p style="font-size: 0.8rem; color: #666; margin: 0;">Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # ==============================================
        # Document Upload Section
        # ==============================================
        st.markdown("### 📁 Documents")
        
        uploaded_files = st.file_uploader(
            "Upload financial documents",
            type=["pdf", "txt", "csv", "docx", "md"],
            accept_multiple_files=True,
            help="Supported: PDF, TXT, CSV, DOCX, MD",
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("📤 Process Documents", type="primary", use_container_width=True):
                on_file_upload(uploaded_files)
        
        # ==============================================
        # Document List
        # ==============================================
        if documents:
            st.markdown("#### Indexed Documents")
            
            for doc in documents:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Truncate long names
                    display_name = doc if len(doc) <= 20 else doc[:17] + "..."
                    st.markdown(f"📄 `{display_name}`")
                with col2:
                    if on_delete_document:
                        if st.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                            on_delete_document(doc)
        else:
            st.info("No documents uploaded yet")
        
        # ==============================================
        # Collection Stats
        # ==============================================
        if collection_stats:
            st.markdown("#### 📊 Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", len(documents))
            with col2:
                st.metric("Chunks", collection_stats.get("document_count", 0))
        
        st.divider()
        
        # ==============================================
        # Settings Section
        # ==============================================
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        model = st.selectbox(
            "LLM Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Select the language model for generation"
        )
        
        # Retrieval settings
        top_k = st.slider(
            "Results to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant chunks to retrieve"
        )
        
        # Advanced settings in expander
        with st.expander("Advanced Settings"):
            enable_reranking = st.checkbox(
                "Enable Reranking",
                value=True,
                help="Use cross-encoder to rerank results"
            )
            
            enable_hybrid = st.checkbox(
                "Hybrid Search",
                value=True,
                help="Combine vector + keyword search"
            )
            
            enable_rewrite = st.checkbox(
                "Query Rewriting",
                value=True,
                help="Optimize queries for better retrieval"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
        
        st.divider()
        
        # ==============================================
        # Info Section
        # ==============================================
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666;">
        <p><strong>Version:</strong> 1.0.0</p>
        <p>RAG-powered financial document assistant with source citations.</p>
        <p><a href="https://github.com/yourusername/financial-knowledge-assistant" target="_blank">GitHub</a> | 
        <a href="/docs" target="_blank">API Docs</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Return current settings
        return {
            "model": model,
            "top_k": top_k,
            "enable_reranking": enable_reranking,
            "enable_hybrid": enable_hybrid,
            "enable_rewrite": enable_rewrite,
            "temperature": temperature,
        }
