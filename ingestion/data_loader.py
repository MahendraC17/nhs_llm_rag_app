# --------------------------------------------------------------------------------
# Data Loading Layer
# Loading and caching processed document chunks for retrieval usage
# --------------------------------------------------------------------------------

import nltk
import streamlit as st

from ingestion.data_chunking import load_and_chunk_pdfs
from config import FAISS_DIR


# Ensuring required tokenizer is available before chunking
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")


# --------------------------------------------------------------------------------
# Document Loader Entry Point
# Loading chunked documents once and caching them to avoid repeated processing
# --------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_documents():
    docs = load_and_chunk_pdfs()

    print(f"FAISS index path: {FAISS_DIR}")
    print("Loading documents...")

    return docs