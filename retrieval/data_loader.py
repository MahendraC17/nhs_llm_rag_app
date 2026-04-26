import nltk

from data_chunking import load_and_chunk_pdfs
from config import FAISS_DIR

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

_docs_cache = None

def get_documents():
    global _docs_cache

    if _docs_cache is None:
        _docs_cache = load_and_chunk_pdfs()
        print(f"FAISS index path: {FAISS_DIR}")
        print("Loading documents...")

    return _docs_cache