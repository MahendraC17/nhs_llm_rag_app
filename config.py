import os
import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-nano-2025-04-14"

FAISS_DIR = "faiss_nhs_sections"
DATA_DIR = "data"