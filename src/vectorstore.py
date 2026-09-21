import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import Embeddings

class Faissvectorstore:
    def __init__(self, persist_dir= "faiss_store", embedding_model= "all-MiniLM-L6-v2", chunk_size = 1000, chunk_overlap = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

    def build_document(self, documents):
        emb_pipe = Embeddings(chunk_overlap=self.chunk_overlap,chunk_size=self.chunk_size,model_name=self.embedding_model)
        chunks = emb_pipe.chunk_document(documents)
        embeddings = emb_pipe.embedding_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'),metadatas)
        self.save()

    def add_embeddings(self, embeddings, metadatas):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_embedding, top_k= 6):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text, top_k = 6):
        print(f"Finding answer for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)