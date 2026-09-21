--> A live Financial RAG app to answer you questions
An enterprise-grade, retrieval-augmented generation engine built to parse, synthesize, and answer complex B2B payment compliance queries across bank MITCs (SBI, HDFC) and RBI regulatory frameworks in real time.

--> Try the interactive assistant live on Streamlit Cloud:
--->  https://financial-rag-v3.streamlit.app/

--> KEY HIGHLIGHTS
Fintech payment aggregators operate under strict regulatory constraints and intricate bank fee structures. Generic LLMs hallucinate non-existent fee rates or fail multi-document comparison.
This engine solves that by combining dense vector search with grounded LLM reasoning:

1.Multi-Hop Synthesis: Cross-references SBI vs. HDFC card policies in a single query pass.

2.Regulatory Guardrails: Enforces RBI Payment Aggregator frameworks, e-Mandate caps, and AFA thresholds.

3.Mathematical Precision: Calculates percentage slabs, fixed surcharges, and 18% GST modifiers.

4.Source Traceability: Appends precise clause tags (e.g., 【Escrow Rules – 4.9】) to eliminate hallucinations.

--> Local setup
# 1. Clone the repository
git clone https://github.com/AtharvaKhare624/Financial-RAG.git
cd Financial-RAG

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (.env file)
GROQ_API_KEY=your_groq_api_key_here

# 5. Run the Streamlit web application
streamlit run app_streamlit.py
