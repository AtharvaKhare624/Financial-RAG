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

# Display example questions for user inspiration (Collapsible to save space)
with st.expander("💡 See example questions you can ask"):
    st.markdown("""
    - **Fee Comparison:** Compare the late payment penalty charged by SBI versus HDFC for an outstanding balance of ₹45,000.
    - **Rent & Utilities:** I need to pay ₹1.5 lakh for office rent. Compare the exact processing fees if I use an SBI card versus an HDFC card.
    - **RBI Mandates:** What are the RBI compliance requirements and e-mandate rules for recurring vendor payments?
    - **Authentication (AFA):** What is the maximum amount I can pay for my business utilities without needing an OTP?
    - **EMI Transfers:** How much extra will HDFC charge me if I transfer a ₹50,000 EMI balance?
    - **Regulatory:** What are the escrow and nodal account rules for payment aggregators according to the RBI?
    - **Over-limit Fees:** If a transaction pushes me over my credit limit, what is the penalty structure?
    - **Definitions:** What is the exact definition of "net-worth" for payment system operators under the RBI framework?
    """)

# Search input box
query = st.text_input("Enter your question:")

if st.button("Analyze Query", type="primary"):
    if query.strip():
        with st.spinner("Searching vector store and generating synthesis..."):
            try:
                # Note: Updated top_k to 6 to handle multi-document comparisons better
                response = rag_engine.search_and_summarize(query, top_k=6)
                st.markdown("### Response")
                st.markdown(response)
            except Exception as e:
                st.error(f"Error processing request: {e}")
    else:
        st.warning("Please enter a valid query.")