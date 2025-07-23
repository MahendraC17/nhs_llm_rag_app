# --------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------

import streamlit as st

# Secret key added in streamlit secret vault
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-nano-2025-04-14"

FAISS_DIR = "faiss_nhs_sections"
DATA_DIR = "data"