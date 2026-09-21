from src.document_loader import document_loader
from src.vectorstore import Faissvectorstore



if __name__=="__main__":
    #docs = document_loader("data")
    store = Faissvectorstore("faiss_store")
    #store.build_document(docs)
    store.load()
    print(store.query("What are the RBI compliance requirements and mandate rules when using corporate credit cards for recurring vendor payments or SaaS billing?",top_k=3))