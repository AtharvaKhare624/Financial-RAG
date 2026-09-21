import os
from dotenv import load_dotenv
from src.search import RAGSearch


load_dotenv()

def main():    

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is missing from your .env file!")
        return

    rag_engine = RAGSearch(
        persist_dir="notebook/vectorstore/db_faiss", 
        embedding_model="all-MiniLM-L6-v2",
        llm_model="openai/gpt-oss-120b"
    )

    query = input("enter your question:")
    print(f"\nUSER QUERY: {query}\n")
    print("Generating answer with LLM")
    
    response = rag_engine.search_and_summarize(query, top_k=4)

    print("\nRESPONSE")
    print(response)

if __name__ == "__main__":
    main()