import os

OPENAI_API_KEY = os.getenv("openai_api_key")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-nano-2025-04-14"

FAISS_DIR = "faiss_nhs_sections"
PDF_DIR = "data"