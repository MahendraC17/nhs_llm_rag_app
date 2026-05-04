# --------------------------------------------------------------------------------
# DATA INGESTION AND CHUNKING PIPELINE
# --------------------------------------------------------------------------------
# Reading NHS PDFs, extracting text with embedded links, splitting into structured
# sections, and converting into chunked documents with metadata for retrieval
# --------------------------------------------------------------------------------

import os
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import NLTKTextSplitter

from config import DATA_DIR

# --------------------------------------------------------------------------------
# EXTRACTING TEXT WITH EMBEDDED LINKS FROM PDF
# --------------------------------------------------------------------------------
# Extracting text from each PDF page while preserving embedded hyperlinks
# Replacing anchor text with text + URL format to retain reference context
# Returning a single combined text string per document
# --------------------------------------------------------------------------------

def text_with_embedded_links(pdf_path):
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

    return full_text


# --------------------------------------------------------------------------------
# SECTION DETECTION USING FLEXIBLE KEYWORD MATCHING
# --------------------------------------------------------------------------------
# Splitting raw text into logical sections
# Mapping variations of headings into canonical section labels
# Falling back to 'general' when no section match is detected
# --------------------------------------------------------------------------------

def split_into_sections(text):

    SECTION_MAP = {
        "symptoms": ["symptom", "sign"],
        "causes": ["cause", "risk"],
        "treatment": ["treat", "therapy", "manage"],
        "prevention": ["prevent"],
        "complications": ["complication"],
        "emergency": ["urgent", "immediate", "when to", "seek help", "see a gp"]
    }

    sections = {}
    current_section = "general"
    sections[current_section] = []

    lines = text.split("\n")

    for line in lines:
        line_clean = line.strip()

        if not line_clean:
            continue

        line_lower = line_clean.lower()
        matched_section = None

        for section, keywords in SECTION_MAP.items():
            for keyword in keywords:
                if keyword in line_lower and len(line_clean) < 80:
                    matched_section = section
                    break
            if matched_section:
                break

        if matched_section:
            current_section = matched_section
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections[current_section].append(line_clean)

    return sections


# --------------------------------------------------------------------------------
# SECTION AWARE CHUNKING USING NLTK SPLITTER
# --------------------------------------------------------------------------------
# Splitting each section into chunks while preserving semantic grouping
# Applying sentence based chunking with overlap to maintain context continuity
# Attaching metadata including disease, source, and section for each chunk
# --------------------------------------------------------------------------------

def sentence_chunk_document(text, disease_name, source_pdf):
    splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=50)

    sections = split_into_sections(text)

    chunks = []

    for section_name, lines in sections.items():
        section_text = "\n".join(lines)
        split_texts = splitter.split_text(section_text)

        for chunk in split_texts:
            clean_chunk = chunk.strip()

            chunks.append(Document(
                page_content=clean_chunk,
                metadata={
                    "disease": disease_name,
                    "source": source_pdf,
                    "section": section_name
                }
            ))

    return chunks


# --------------------------------------------------------------------------------
# DEDUPLICATING CHUNKS
# --------------------------------------------------------------------------------
# Removing duplicate text chunks to avoid redundant embeddings and retrieval noise
# Reducing index size and improving retrieval efficiency
# --------------------------------------------------------------------------------

def deduplicate_chunks(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        text = doc.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_docs.append(doc)

    return unique_docs


# --------------------------------------------------------------------------------
# LOADING PDF DATA AND BUILDING CHUNK COLLECTION
# --------------------------------------------------------------------------------
# Iterating through all PDFs in the data source
# Extracting text, applying section-aware chunking, and aggregating results
# Performing final deduplication before returning documents for embedding
# --------------------------------------------------------------------------------

def load_and_chunk_pdfs(pdf_folder=DATA_DIR):
    all_chunks = []

    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            disease_name = file_name.replace("_", " ").replace(".pdf", "")

            full_text = text_with_embedded_links(
                os.path.join(pdf_folder, file_name)
            )

            chunks = sentence_chunk_document(
                full_text,
                disease_name,
                f"{disease_name}.pdf"
            )

            all_chunks.extend(chunks)

    return deduplicate_chunks(all_chunks)