from src.document_loader import document_loader
from src.embedding import Embeddings



if __name__=="__main__":
    docs = document_loader("data")
    E1 = Embeddings()
    chunks = E1.chunk_document(docs)
    chunk_vectors = E1.embedding_chunks(chunks)
    print(chunk_vectors)
    