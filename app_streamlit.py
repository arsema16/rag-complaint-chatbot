"""
Streamlit UI for RAG Chatbot
"""

import streamlit as st
from src.rag_pipeline import RAGPipeline
import pandas as pd

# Page config
st.set_page_config(
    page_title="Complaint Analysis RAG",
    page_icon="🏦",
    layout="wide"
)

# Initialize RAG
@st.cache_resource
def load_rag():
    return RAGPipeline()

st.title("🏦 CrediTrust Financial - Complaint Analysis")
st.markdown("Ask questions about customer complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of sources (k)", min_value=1, max_value=10, value=3)
    st.markdown("---")
    st.markdown("### 📊 Stats")
    st.markdown(f"- **Total Chunks:** 400")
    st.markdown(f"- **Products:** 4")
    st.markdown(f"- **k-value:** {k}")

# Main area
question = st.text_area(
    "💬 Ask a question:",
    placeholder="e.g., Why are customers unhappy with credit card fees?",
    height=100
)

col1, col2 = st.columns([1, 5])

with col1:
    if st.button("🔍 Ask", use_container_width=True):
        if question:
            with st.spinner("🔍 Searching for answers..."):
                rag = load_rag()
                result = rag.query(question, k=k)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(f"**{result['answer']}**")
                
                # Display sources
                st.markdown("### 📚 Sources")
                if result['sources']:
                    for i, source in enumerate(result['sources']):
                        with st.expander(f"Source {i+1} - {source['metadata']['product_category']} (Score: {source['score']:.3f})"):
                            st.markdown(f"**Complaint ID:** {source['metadata']['complaint_id']}")
                            st.markdown(f"**Issue:** {source['metadata'].get('issue', 'N/A')}")
                            st.markdown(f"**Text:** {source['text']}")
                else:
                    st.info("No sources found.")
        else:
            st.warning("Please enter a question.")

with col2:
    st.markdown("### 📋 Example Questions")
    examples = [
        "Why are customers unhappy with credit card fees?",
        "What are the main issues with personal loans?",
        "Tell me about savings account complaints",
        "What problems do people have with money transfers?"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.question = ex
            st.rerun()