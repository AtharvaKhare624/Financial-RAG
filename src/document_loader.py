from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyMuPDFLoader

def document_loader(data_dir):
    path = Path(data_dir).resolve()

    txt_files = list(path.glob("**/*.txt"))

    documents = []

    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = txt_file.name
                doc.metadata["file_type"] = "text file"
            documents.extend(docs)
        except Exception as e:
            print(f"error :{e}")

    return documents



"""    path_pdf = Path(data_dir).resolve()
    pdf_files = list(path_pdf.glob("*/**.pdf"))

    for pdf_file in pdf_files:
        
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf file"
            documents.extend(docs)
        except Exception as e:
            print(f"error:{e}")"""
