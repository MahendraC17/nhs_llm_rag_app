import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-nano-2025-04-14"

FAISS_DIR = "faiss_nhs_sections"
PDF_DIR = "data"