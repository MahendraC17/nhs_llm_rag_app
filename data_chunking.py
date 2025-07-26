# --------------------------------------------------------------------------------
# Reads all the pdf, splits the data per disease, per pdfs and turns them into chunks
# Returns all the chunks for further embedding in -> rag_pipeline.py
# --------------------------------------------------------------------------------

import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import NLTKTextSplitter
from config import DATA_DIR

def deduplicate_chunks(docs):
    """
    To not to repeat chunking process and store chunks which are already been made,
    this function ensures it doesn't. Saves time and resources to embeded those chunks.
    """
    seen = set()
    unique_docs = []
    for doc in docs:
        text = doc.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_docs.append(doc)
    return unique_docs

def sentence_chunk_document(text, disease_name, source_pdf):
    """
    Sentence-level chunking using NLTK splitter.
    Preserving disease and source info in metadata.
    """
    splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = splitter.split_text(text)

    chunks = []
    for chunk in split_texts:
        chunks.append(Document(
            page_content=chunk.strip(),
            metadata={
                "disease": disease_name,
                "source": source_pdf
            }
        ))
    return chunks

def load_and_chunk_pdfs(pdf_folder=DATA_DIR):
    """
    Loads PDFs and returns sentence-based chunks with source metadata.

    Args:
        DATA_DIR (str): Path to the folder containing PDF files.

    Returns:
        List[Document]: A list of unique, deduplicated text chunks 
        ready for embedding in the RAG pipeline.
    """
    all_chunks = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            disease_name = file_name.replace("_", " ").replace(".pdf", "")
            loader = PyPDFLoader(os.path.join(pdf_folder, file_name))
            docs = loader.load()

            full_text = "\n".join([doc.page_content for doc in docs])
            chunks = sentence_chunk_document(full_text, disease_name, f"{disease_name}.pdf")
            all_chunks.extend(chunks)

    return deduplicate_chunks(all_chunks)