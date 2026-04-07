"""
Chat Component

Main chat interface with message display and input.
"""

import streamlit as st
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Optional


@dataclass
class ChatMessage:
    """Chat message data class."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: Optional[list[dict]] = None
    confidence: Optional[str] = None
    latency_ms: Optional[float] = None


def render_chat_message(message: ChatMessage) -> None:
    """Render a single chat message with styling."""
    
    if message.role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
            
            # Show metadata for assistant messages
            if message.sources or message.confidence or message.latency_ms:
                render_message_footer(message)


def render_message_footer(message: ChatMessage) -> None:
    """Render the footer with sources and metadata."""
    
    # Confidence and latency
    footer_parts = []
    
    if message.confidence:
        confidence_emoji = {
            "HIGH": "🟢",
            "MEDIUM": "🟡", 
            "LOW": "🔴"
        }.get(message.confidence, "⚪")
        footer_parts.append(f"{confidence_emoji} {message.confidence}")
    
    if message.latency_ms:
        footer_parts.append(f"⏱️ {message.latency_ms:.0f}ms")
    
    if message.sources:
        footer_parts.append(f"📚 {len(message.sources)} sources")
    
    if footer_parts:
        st.caption(" | ".join(footer_parts))


def render_source_cards(sources: list[dict]) -> None:
    """Render source document cards."""
    
    if not sources:
        return
    
    with st.expander(f"📚 View Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{source.get('source_file', 'Unknown')}**")
                    st.caption(f"Page {source.get('page_number', 'N/A')}")
                
                with col2:
                    score = source.get('score', 0)
                    if score > 0:
                        st.metric("Relevance", f"{score:.0%}")
                
                # Preview
                preview = source.get('preview', '')
                if preview:
                    st.text_area(
                        "Preview",
                        value=preview,
                        height=80,
                        disabled=True,
                        key=f"source_preview_{i}",
                        label_visibility="collapsed"
                    )
                
                st.divider()


def render_chat_input(
    placeholder: str = "Ask a question about your documents...",
    key: str = "chat_input"
) -> Optional[str]:
    """Render the chat input field."""
    return st.chat_input(placeholder, key=key)


def render_suggested_questions(
    questions: list[str],
    on_select: callable
) -> None:
    """Render suggested question chips."""
    
    st.markdown("**💡 Try asking:**")
    
    cols = st.columns(min(len(questions), 3))
    for i, question in enumerate(questions[:3]):
        with cols[i]:
            if st.button(
                question[:40] + "..." if len(question) > 40 else question,
                key=f"suggested_{i}",
                use_container_width=True
            ):
                on_select(question)


def render_empty_state() -> None:
    """Render the empty chat state."""
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">🏦</h1>
        <h2 style="color: #333; margin-bottom: 0.5rem;">Financial Knowledge Assistant</h2>
        <p style="color: #666; font-size: 1rem;">
            Upload financial documents and ask questions to get cited answers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <p style="font-size: 2rem; margin: 0;">📄</p>
            <p style="font-weight: bold; margin: 0.5rem 0;">Multi-Format</p>
            <p style="font-size: 0.8rem; color: #666; margin: 0;">PDF, TXT, CSV, DOCX</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <p style="font-size: 2rem; margin: 0;">🔍</p>
            <p style="font-weight: bold; margin: 0.5rem 0;">Smart Search</p>
            <p style="font-size: 0.8rem; color: #666; margin: 0;">Hybrid retrieval + reranking</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <p style="font-size: 2rem; margin: 0;">📚</p>
            <p style="font-weight: bold; margin: 0.5rem 0;">Cited Answers</p>
            <p style="font-size: 0.8rem; color: #666; margin: 0;">Every answer with sources</p>
        </div>
        """, unsafe_allow_html=True)


def render_processing_indicator(message: str = "Processing...") -> None:
    """Render a processing indicator."""
    with st.spinner(message):
        st.empty()


def stream_response(
    response_generator: Iterator[str],
    placeholder: st.empty
) -> str:
    """Stream response chunks to a placeholder."""
    full_response = ""
    
    for chunk in response_generator:
        full_response += chunk
        placeholder.markdown(full_response + "▌")
    
    placeholder.markdown(full_response)
    return full_response
