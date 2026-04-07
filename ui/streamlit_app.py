"""
Financial Knowledge Assistant - Streamlit UI
==============================================

A professional RAG-powered interface for querying financial documents
with source citations, confidence scoring, and evaluation metrics.

Run with: streamlit run ui/streamlit_app.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import (
    DocumentParsingError,
    NoRelevantContextError,
    RAGError,
)

logger = get_logger(__name__)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Financial Knowledge Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/financial-knowledge-assistant",
        "Report a bug": "https://github.com/yourusername/financial-knowledge-assistant/issues",
        "About": """
        ## Financial Knowledge Assistant
        
        A production-ready RAG system for financial document analysis.
        
        **Features:**
        - 📄 Multi-format document support
        - 🔍 Hybrid search with reranking
        - 📚 Source citations
        - ✅ Confidence scoring
        """
    }
)

# =============================================================================
# Load Custom CSS
# =============================================================================

def load_css():
    """Load custom CSS styles."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =============================================================================
# Initialize Session State
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "model": "gpt-4o-mini",
            "top_k": 5,
            "enable_reranking": True,
            "enable_hybrid": True,
            "enable_rewrite": True,
            "temperature": 0.1,
        }

init_session_state()

# =============================================================================
# Lazy Loading of Components
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_rag_components():
    """Load RAG components (cached)."""
    try:
        from app.retrieval.embeddings import EmbeddingModel
        from app.retrieval.vector_store import VectorStore
        from app.retrieval.retriever import Retriever
        from app.generation.chain import RAGChain
        
        # Initialize embeddings
        embeddings = EmbeddingModel()
        
        # Initialize vector store
        vector_store = VectorStore(embeddings=embeddings)
        
        # Initialize retriever
        retriever = Retriever(vector_store=vector_store)
        
        # Initialize RAG chain
        rag_chain = RAGChain(retriever=retriever)
        
        return {
            "embeddings": embeddings,
            "vector_store": vector_store,
            "retriever": retriever,
            "rag_chain": rag_chain,
        }
    except Exception as e:
        logger.error(f"Failed to load RAG components: {e}")
        return None

# =============================================================================
# Document Processing
# =============================================================================

def process_uploaded_files(files: list) -> tuple[int, list[str]]:
    """
    Process uploaded files and add to vector store.
    
    Returns:
        Tuple of (success_count, error_messages)
    """
    from app.ingestion.document_loader import DocumentLoader
    from app.ingestion.chunker import DocumentChunker
    from app.ingestion.preprocessor import TextPreprocessor
    
    loader = DocumentLoader()
    chunker = DocumentChunker()
    preprocessor = TextPreprocessor()
    
    success_count = 0
    errors = []
    
    # Get or initialize components
    components = load_rag_components()
    if not components:
        return 0, ["Failed to initialize RAG components"]
    
    vector_store = components["vector_store"]
    
    for file in files:
        try:
            # Save temp file
            temp_dir = Path("./data/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / file.name
            
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Load and process
            documents = loader.load(temp_path)
            
            # Preprocess
            for doc in documents:
                doc.page_content = preprocessor.preprocess(doc.page_content)
            
            # Chunk
            chunks = chunker.chunk_documents(documents)
            
            # Add to vector store
            vector_store.add_documents(chunks)
            
            # Track
            st.session_state.documents.append(file.name)
            success_count += 1
            
            logger.info(f"Processed {file.name}: {len(chunks)} chunks")
            
        except DocumentParsingError as e:
            errors.append(f"{file.name}: {str(e)}")
            logger.error(f"Failed to process {file.name}: {e}")
        except Exception as e:
            errors.append(f"{file.name}: Unexpected error - {str(e)}")
            logger.exception(f"Unexpected error processing {file.name}")
    
    return success_count, errors

def delete_document(filename: str) -> bool:
    """Delete a document from the vector store."""
    try:
        components = load_rag_components()
        if not components:
            return False
        
        vector_store = components["vector_store"]
        vector_store.delete_by_source(filename)
        
        if filename in st.session_state.documents:
            st.session_state.documents.remove(filename)
        
        return True
    except Exception as e:
        logger.error(f"Failed to delete {filename}: {e}")
        return False

# =============================================================================
# Query Processing
# =============================================================================

def process_query(question: str) -> dict[str, Any]:
    """
    Process a user query and return the response with sources.
    
    Returns:
        Dictionary with answer, sources, confidence, and latency
    """
    start_time = time.time()
    
    components = load_rag_components()
    if not components:
        return {
            "answer": "⚠️ System not initialized. Please check logs.",
            "sources": [],
            "confidence": "LOW",
            "latency_ms": 0,
        }
    
    rag_chain = components["rag_chain"]
    
    try:
        # Get settings
        current_settings = st.session_state.settings
        
        # Query
        response = rag_chain.query(
            question=question,
            top_k=current_settings["top_k"],
            enable_reranking=current_settings["enable_reranking"],
            enable_hybrid=current_settings["enable_hybrid"],
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Format sources
        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                "source_file": doc.metadata.get("source", "Unknown"),
                "page_number": doc.metadata.get("page", "N/A"),
                "score": doc.metadata.get("score", 0),
                "preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "chunk_type": doc.metadata.get("chunk_type", "text"),
            })
        
        # Record in history
        st.session_state.query_history.append({
            "question": question,
            "latency_ms": latency_ms,
            "confidence": response.get("confidence", "MEDIUM"),
            "num_sources": len(sources),
            "timestamp": datetime.now().isoformat(),
        })
        
        return {
            "answer": response.get("answer", "No answer generated."),
            "sources": sources,
            "confidence": response.get("confidence", "MEDIUM"),
            "latency_ms": latency_ms,
        }
        
    except NoRelevantContextError:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing or upload more relevant documents.",
            "sources": [],
            "confidence": "LOW",
            "latency_ms": (time.time() - start_time) * 1000,
        }
    except RAGError as e:
        return {
            "answer": f"⚠️ An error occurred: {str(e)}",
            "sources": [],
            "confidence": "LOW",
            "latency_ms": (time.time() - start_time) * 1000,
        }
    except Exception as e:
        logger.exception("Unexpected error during query")
        return {
            "answer": f"⚠️ An unexpected error occurred. Please try again.",
            "sources": [],
            "confidence": "LOW",
            "latency_ms": (time.time() - start_time) * 1000,
        }

# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render the sidebar with document management and settings."""
    
    with st.sidebar:
        # Logo and branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <span style="font-size: 3rem;">🏦</span>
            <h1 style="font-size: 1.25rem; margin: 0.5rem 0 0 0; font-weight: 700;">
                Financial Knowledge
            </h1>
            <p style="font-size: 0.85rem; color: #666; margin: 0;">
                Assistant
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # ===========================================
        # Document Upload Section
        # ===========================================
        st.markdown("### 📁 Documents")
        
        uploaded_files = st.file_uploader(
            "Upload financial documents",
            type=["pdf", "txt", "csv", "docx", "md"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, CSV, DOCX, Markdown",
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("📤 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    success, errors = process_uploaded_files(uploaded_files)
                    
                    if success > 0:
                        st.success(f"✅ Processed {success} document(s)")
                    if errors:
                        for error in errors:
                            st.error(error)
        
        # ===========================================
        # Document List
        # ===========================================
        if st.session_state.documents:
            st.markdown("#### 📚 Indexed Documents")
            
            for doc in st.session_state.documents:
                col1, col2 = st.columns([4, 1])
                with col1:
                    display_name = doc if len(doc) <= 20 else doc[:17] + "..."
                    st.markdown(f"📄 `{display_name}`")
                with col2:
                    if st.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                        if delete_document(doc):
                            st.rerun()
        else:
            st.info("📭 No documents uploaded yet")
        
        # ===========================================
        # Collection Stats
        # ===========================================
        components = load_rag_components()
        if components:
            try:
                stats = components["vector_store"].get_stats()
                if stats.get("document_count", 0) > 0:
                    st.markdown("#### 📊 Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Files", len(st.session_state.documents))
                    with col2:
                        st.metric("Chunks", stats.get("document_count", 0))
            except Exception:
                pass
        
        st.divider()
        
        # ===========================================
        # Settings
        # ===========================================
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        st.session_state.settings["model"] = st.selectbox(
            "LLM Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Select the language model for generation"
        )
        
        # Retrieval settings
        st.session_state.settings["top_k"] = st.slider(
            "Results to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant chunks to retrieve"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced"):
            st.session_state.settings["enable_reranking"] = st.checkbox(
                "Enable Reranking",
                value=True,
                help="Use cross-encoder to rerank results"
            )
            
            st.session_state.settings["enable_hybrid"] = st.checkbox(
                "Hybrid Search",
                value=True,
                help="Combine vector + keyword search"
            )
            
            st.session_state.settings["enable_rewrite"] = st.checkbox(
                "Query Rewriting",
                value=True,
                help="Optimize queries for better retrieval"
            )
            
            st.session_state.settings["temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
        
        st.divider()
        
        # ===========================================
        # Actions
        # ===========================================
        st.markdown("### 🔄 Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("📊 Metrics", use_container_width=True):
                st.session_state.show_metrics = True
        
        st.divider()
        
        # ===========================================
        # About
        # ===========================================
        st.markdown("### ℹ️ About")
        st.caption("""
        **Version:** 1.0.0
        
        RAG-powered financial document analysis with source citations.
        
        [GitHub](https://github.com/yourusername/financial-knowledge-assistant) | 
        [API Docs](/docs)
        """)

# =============================================================================
# Main Chat Interface
# =============================================================================

def render_chat():
    """Render the main chat interface."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 1.75rem; font-weight: 700; margin: 0;">
            💬 Chat with Your Documents
        </h1>
        <p style="color: #666; margin: 0.5rem 0 0 0;">
            Ask questions about your financial documents
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show empty state if no messages
    if not st.session_state.messages:
        render_empty_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        render_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
        # Add user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now(),
        }
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Process and display response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = process_query(prompt)
            
            # Display answer
            st.markdown(response["answer"])
            
            # Display metadata
            render_response_metadata(response)
            
            # Display sources
            if response["sources"]:
                render_sources_expander(response["sources"])
        
        # Save assistant message
        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
            "timestamp": datetime.now(),
            "sources": response["sources"],
            "confidence": response["confidence"],
            "latency_ms": response["latency_ms"],
        }
        st.session_state.messages.append(assistant_message)

def render_message(message: dict):
    """Render a single chat message."""
    
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])
            
            if message.get("confidence") or message.get("latency_ms"):
                render_response_metadata(message)
            
            if message.get("sources"):
                render_sources_expander(message["sources"])

def render_response_metadata(response: dict):
    """Render response metadata (confidence, latency)."""
    
    confidence = response.get("confidence", "MEDIUM")
    latency = response.get("latency_ms", 0)
    sources_count = len(response.get("sources", []))
    
    # Confidence badge
    confidence_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")
    confidence_color = {"HIGH": "#22c55e", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}.get(confidence, "#666")
    
    st.markdown(f"""
    <div style="display: flex; gap: 1rem; align-items: center; margin-top: 0.5rem; font-size: 0.85rem; color: #666;">
        <span style="background: {confidence_color}15; color: {confidence_color}; padding: 2px 8px; border-radius: 4px; font-weight: 500;">
            {confidence_emoji} {confidence} Confidence
        </span>
        <span>⏱️ {latency:.0f}ms</span>
        <span>📚 {sources_count} sources</span>
    </div>
    """, unsafe_allow_html=True)

def render_sources_expander(sources: list[dict]):
    """Render expandable source citations."""
    
    with st.expander(f"📚 View Source Citations ({len(sources)})", expanded=False):
        for i, source in enumerate(sources):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    filename = source.get("source_file", "Unknown")
                    page = source.get("page_number", "N/A")
                    chunk_type = source.get("chunk_type", "text")
                    
                    # File icon
                    ext = filename.split(".")[-1].lower() if "." in filename else ""
                    icon = {"pdf": "📕", "txt": "📄", "csv": "📊", "docx": "📘"}.get(ext, "📄")
                    
                    st.markdown(f"**{icon} {filename}** (Page {page})")
                    st.caption(f"Type: {chunk_type}")
                
                with col2:
                    score = source.get("score", 0)
                    if score > 0:
                        color = "#22c55e" if score > 0.8 else "#f59e0b" if score > 0.6 else "#ef4444"
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 1.25rem; font-weight: 700; color: {color};">
                                {score:.0%}
                            </div>
                            <div style="font-size: 0.75rem; color: #666;">Match</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Preview
                preview = source.get("preview", "")
                if preview:
                    st.text_area(
                        "Preview",
                        value=preview,
                        height=80,
                        disabled=True,
                        key=f"source_{i}",
                        label_visibility="collapsed"
                    )
                
                if i < len(sources) - 1:
                    st.divider()

def render_empty_state():
    """Render the empty chat state with feature highlights."""
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem;">
        <p style="font-size: 4rem; margin-bottom: 1rem;">🏦</p>
        <h2 style="color: #333; margin-bottom: 0.5rem; font-weight: 600;">
            Welcome to Financial Knowledge Assistant
        </h2>
        <p style="color: #666; font-size: 1rem; max-width: 600px; margin: 0 auto;">
            Upload your financial documents (10-K, earnings reports, etc.) and ask questions 
            to get AI-powered answers with source citations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("📄", "Multi-Format Support", "PDF, TXT, CSV, DOCX files with intelligent table extraction"),
        ("🔍", "Hybrid Search", "Vector + BM25 keyword search with cross-encoder reranking"),
        ("📚", "Source Citations", "Every answer backed by specific document references"),
    ]
    
    for col, (emoji, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
            <div style="
                background: #f8fafc;
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
                height: 180px;
                border: 1px solid #e2e8f0;
            ">
                <p style="font-size: 2.5rem; margin: 0;">{emoji}</p>
                <p style="font-weight: 600; margin: 0.75rem 0 0.5rem 0; color: #333;">{title}</p>
                <p style="font-size: 0.85rem; color: #666; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Sample questions
    st.markdown("### 💡 Sample Questions")
    
    sample_questions = [
        "What was the total revenue for FY2023?",
        "What are the main risk factors mentioned?",
        "Summarize the company's business strategy",
        "What were the key highlights from the earnings call?",
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(f"💬 {question}", key=f"sample_{i}", use_container_width=True):
                st.session_state.sample_question = question
                st.rerun()
    
    # Handle sample question selection
    if "sample_question" in st.session_state:
        question = st.session_state.sample_question
        del st.session_state.sample_question
        
        # Add to messages and process
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now(),
        })
        st.rerun()

# =============================================================================
# Metrics Modal
# =============================================================================

def render_metrics_view():
    """Render the metrics/analytics view."""
    
    if not st.session_state.get("show_metrics"):
        return
    
    st.markdown("## 📊 Performance Metrics")
    
    history = st.session_state.query_history
    
    if not history:
        st.info("No query history yet. Start asking questions!")
        if st.button("Close"):
            st.session_state.show_metrics = False
            st.rerun()
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", len(history))
    
    with col2:
        avg_latency = sum(q["latency_ms"] for q in history) / len(history)
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    
    with col3:
        high_conf = sum(1 for q in history if q["confidence"] == "HIGH")
        st.metric("High Confidence", f"{high_conf}/{len(history)}")
    
    with col4:
        avg_sources = sum(q["num_sources"] for q in history) / len(history)
        st.metric("Avg Sources", f"{avg_sources:.1f}")
    
    # Recent queries table
    st.markdown("### Recent Queries")
    
    for query in reversed(history[-10:]):
        col1, col2, col3 = st.columns([4, 1, 1])
        
        with col1:
            q = query["question"]
            st.write(q[:60] + "..." if len(q) > 60 else q)
        with col2:
            conf = query["confidence"]
            emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf, "⚪")
            st.write(f"{emoji} {conf}")
        with col3:
            st.write(f"{query['latency_ms']:.0f}ms")
    
    if st.button("Close Metrics"):
        st.session_state.show_metrics = False
        st.rerun()

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main application entry point."""
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if st.session_state.get("show_metrics"):
        render_metrics_view()
    else:
        render_chat()

if __name__ == "__main__":
    main()
