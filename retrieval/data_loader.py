from data_chunking import load_and_chunk_pdfs
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

_docs_cache = None

def get_documents():
    global _docs_cache

    if _docs_cache is None:
        _docs_cache = load_and_chunk_pdfs()
        print("Loading documents...")

    return _docs_cache