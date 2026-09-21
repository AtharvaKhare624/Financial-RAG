import os
from dotenv import load_dotenv
from src.vectorstore import Faissvectorstore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir = "faiss_store", embedding_model= "all-MiniLM-L6-v2", llm_model="openai/gpt-oss-120b"):
        self.vectorstore = Faissvectorstore(persist_dir, embedding_model)
        

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.document_loader import document_loader
            docs = document_loader("data")
            self.vectorstore.build_document(docs)
        else:
            self.vectorstore.load()
            
        self.llm = ChatGroq(model_name=llm_model)
        print(f"Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query, top_k= 6):
        results = self.vectorstore.query(query, top_k=top_k)
        
        texts = [
            f"[Document Source: {r['metadata'].get('source', 'Unknown')}]\n" + r["metadata"].get("text", "") 
            for r in results if r["metadata"]
        ]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        
        prompt = f"""You are an expert B2B financial compliance assistant.
Answer the question using strictly the context provided below. If you cannot answer the question using the context, state "I do not have enough information to answer that.": '{query}'

Context:
{context}

Summary:"""
        response = self.llm.invoke([prompt])
        return response.content