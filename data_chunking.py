# --------------------------------------------------------------------------------
# Reads all the pdf, splits the data per disease, per pdfs and turns them into chunks
# Returns all the chunks for further embedding in -> rag_pipeline.py
# --------------------------------------------------------------------------------

import os
import re
import fitz
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import NLTKTextSplitter
from config.settings import Settings

settings = Settings()

def text_with_embedded_links(pdf_path):
    """
    Extracting all the embedded hyperlinks from a word (If embedded)
    and appending the extracted hyperlink at the end of the word.
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        links = page.get_links()

        page_text = page.get_text()

        for link in links:
            if "uri" in link and "from" in link:
                rect = fitz.Rect(link["from"])
                linked_text = page.get_textbox(rect).strip()

                if linked_text:
                    linked_with_url = f"{linked_text} ({link['uri']})"
                    page_text = page_text.replace(linked_text, linked_with_url, 1)

        full_text += page_text + "\n"
    
        # output_path = pdf_path.replace(".pdf", ".txt")
        # with open(output_path, "w", encoding="utf-8") as f:
        #     f.write(full_text)

    return full_text


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

def load_and_chunk_pdfs(pdf_folder=settings.data_dir):
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

            full_text = text_with_embedded_links(os.path.join(pdf_folder, file_name))

            chunks = sentence_chunk_document(full_text, disease_name, f"{disease_name}.pdf")
            all_chunks.extend(chunks)

    return deduplicate_chunks(all_chunks)