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
        You are an AI medical assistant trained strictly on NHS documents related to diseases, symptoms, and treatment.

        Only answer using the information provided in the context below. If the answer is not clearly and directly present, respond with: "I don’t know." Do NOT make assumptions or generate unsupported content.

        Format your response in:
        - Concise bullet points for clarity
        - Include self-care guidance (if relevant)
        - Highlight any red-flag symptoms that require urgent medical attention using [Red Flag] tags
        - Always include the disclaimer at the end

        Context:
        {summaries}

        Question:
        {question}

        Answer:
        """
