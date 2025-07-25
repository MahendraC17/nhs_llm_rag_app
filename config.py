# --------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------

import os
import streamlit as st

# Secret key added in streamlit secret vault
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# Local Key

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-nano-2025-04-14"

FAISS_DIR = "faiss_nhs_sections"
DATA_DIR = "data"

TEMPLATE = """
        You are a medical assistant trained only on NHS disease-related documents.
        If the user asks a question unrelated to this domain, politely respond with:
        "I'm not trained to answer that. Please ask a question related to NHS diseases."

        Context: {summaries}

        Question: {question}
        Answer:"""
