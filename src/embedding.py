import numpy as np
from typing import List,Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.document_loader import document_loader

class Embeddings:
    def __init__(self,model_name = "all-MiniLM-L6-v2", chunk_size = 1000, chunk_overlap = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)

    def chunk_document(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators=["\n\n","\n"," ",""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embedding_chunks(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        print(f"generating embeddigns for {len(texts)} chunks")
        embeddings = self.model.encode(texts, show_progress_bar = True)
        return embeddings