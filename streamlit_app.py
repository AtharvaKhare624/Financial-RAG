import os
import streamlit as st
from dotenv import load_dotenv
from src.search import RAGSearch

# Load environment variables
load_dotenv()

# Set up page layout and title
st.set_page_config(
    page_title="B2B Financial Compliance RAG",
    page_icon="💳",
    layout="centered"
)

st.title("B2B Financial & Regulatory RAG Assistant")
st.markdown("Query credit card policy documents (**SBI/HDFC**) and **RBI compliance frameworks** in real time.")

# Cache the RAG engine in RAM so FAISS vector store is loaded ONCE on app startup
@st.cache_resource
def load_rag_engine():
    return RAGSearch(
        persist_dir="faiss_store", 
        embedding_model="all-MiniLM-L6-v2",
        llm_model="openai/gpt-oss-120b"
    )

# Check API key before loading
if not os.getenv("GROQ_API_KEY"):
    st.error("`GROQ_API_KEY` is missing from environment variables or .env file.")
    st.stop()

rag_engine = load_rag_engine()

# Example prompt shortcuts
st.caption("Try an example question:")
col1, col2 = st.columns(2)
example_query = None

if col1.button("SBI vs HDFC ₹20,000 Fee"):
    example_query = "Compare the late payment penalty charged by SBI versus HDFC for an outstanding balance of ₹20,000. Which card charges higher fees for this slab?"
if col2.button("RBI Recurring E-Mandate Rules"):
    example_query = "What are the RBI compliance requirements and mandate rules when using corporate credit cards for recurring vendor payments or SaaS billing?"

# Search input box
query = st.text_input("Enter your question:", value=example_query if example_query else "")

if st.button("Analyze Query", type="primary"):
    if query.strip():
        with st.spinner("Searching vector store and generating synthesis..."):
            try:
                response = rag_engine.search_and_summarize(query, top_k=4)
                st.markdown("### Response")
                st.markdown(response)
            except Exception as e:
                st.error(f"Error processing request: {e}")
    else:
        st.warning("Please enter a valid query.")