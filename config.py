import streamlit as st
import openai

openai.api_key = st.secrets["openai"]["api_key"]
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-nano-2025-04-14"

FAISS_DIR = "faiss_nhs_sections"
DATA_DIR = "data"