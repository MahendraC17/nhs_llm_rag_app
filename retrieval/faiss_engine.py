# --------------------------------------------------------------------------------
# FAISS ENGINE - Vector Store + Retriever
# --------------------------------------------------------------------------------

import os
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from retrieval.data_loader import get_documents

from config import EMBEDDING_MODEL, FAISS_DIR, OPENAI_API_KEY
FINGERPRINT_FILE = "faiss_nhs_sections/fingerprint.txt"


# --------------------------------------------------------------------------------
# DATA FINGERPRINTING
# --------------------------------------------------------------------------------
# Comparing stored fingerprint with current data state
# Triggering rebuild when mismatch if detected
# --------------------------------------------------------------------------------

def compute_fingerprint(docs):
    combined = "".join([d.page_content for d in docs])
    return hashlib.md5(combined.encode()).hexdigest()


def load_fingerprint():
    if not os.path.exists(FINGERPRINT_FILE):
        return None

    with open(FINGERPRINT_FILE, "r") as f:
        return f.read().strip()


def save_fingerprint(fp):
    os.makedirs(os.path.dirname(FINGERPRINT_FILE), exist_ok=True)

    with open(FINGERPRINT_FILE, "w") as f:
        f.write(fp)


class FAISSEngine:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        self.embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = self._load_or_build_index()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def _load_or_build_index(self):
        docs = get_documents()

        current_fp = compute_fingerprint(docs)
        stored_fp = load_fingerprint()

        print("Current FP:", current_fp)
        print("Stored FP:", stored_fp)

        index_path = f"{FAISS_DIR}/index.faiss"

        if os.path.exists(index_path) and current_fp == stored_fp:
            print("Loading existing FAISS index...")
            vectorstore = FAISS.load_local(
                FAISS_DIR,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            print("Rebuilding FAISS index...")
            vectorstore = FAISS.from_documents(docs, self.embedding_model)
            vectorstore.save_local(FAISS_DIR)
            save_fingerprint(current_fp)

        return vectorstore

    def search(self, query):
        return self.retriever.invoke(query)