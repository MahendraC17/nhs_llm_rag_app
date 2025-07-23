# --------------------------------------------------------------------------------
# Reads all the pdf, splits the data per disease, per pdfs and turns them into chunks
# Returns all the chunks for further embedding in -> rag_pipeline.py
# --------------------------------------------------------------------------------

import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from config import DATA_DIR

# Hard coded section heading, hardcoded for now but might need to overhaul this entire 
# logic as many of the diseases don't follow the same headings
SECTION_HEADERS = [
    "Overview", "Symptoms", "Causes", "Diagnosis", "Treatment",
    "Prevention", "Complications"
]

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

def split_sections(text, disease_name):
    """
    Splitting the PDFs by the given sections headings.
    Storing the metadata -> disease name, section name and source.
    """
    pattern = "|".join([re.escape(sec) for sec in SECTION_HEADERS])
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

    sections = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        raw_title = matches[i].group(0).strip()

        normalized_title = next(
            (s for s in SECTION_HEADERS if s.lower() == raw_title.lower()),
            raw_title
        )

        section_text = text[start:end].strip()

        if section_text:
            sections.append(Document(
                page_content=section_text,
                metadata={
                    "disease": disease_name,
                    "section": normalized_title,
                    "source": f"{disease_name}.pdf"
                }
            ))
    return sections

def load_and_chunk_pdfs(pdf_folder=DATA_DIR):
    """
    Loads all PDF files in the specified folder, extracts their text.
    Splits the content into disease-specific sections, and chunks them.

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
            sectioned_docs = split_sections(full_text, disease_name)
            all_chunks.extend(sectioned_docs)

    unique_chunks = deduplicate_chunks(all_chunks)
    return unique_chunks