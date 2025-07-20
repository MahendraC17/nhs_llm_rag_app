import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document


from config import DATA_DIR

SECTION_HEADERS = [
    "Overview", "Symptoms", "Causes", "Diagnosis", "Treatment",
    "Prevention", "Complications"
]

def split_sections(text, disease_name):

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
                metadata={"disease": disease_name, "section": normalized_title}
            ))
    return sections

def deduplicate_chunks(docs):
    seen = set()
    unique_docs = []
    for doc in docs:
        text = doc.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_docs.append(doc)
    return unique_docs

def split_sections(text, disease_name):
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
