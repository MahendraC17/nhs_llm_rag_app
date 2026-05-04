import nltk
import streamlit as st

from data_chunking import load_and_chunk_pdfs
from config import FAISS_DIR

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

@st.cache_resource(show_spinner=False)
def get_documents():
    docs = load_and_chunk_pdfs()
    print(f"FAISS index path: {FAISS_DIR}")
    print("Loading documents...")
    return docs