# --------------------------------------------------------------------------------
# FAISS ENGINE - Vector Store + Retriever
# --------------------------------------------------------------------------------

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from retrieval.data_loader import get_documents

from config import EMBEDDING_MODEL, FAISS_DIR, OPENAI_API_KEY

class FAISSEngine:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        self.embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = self._load_or_build_index()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def _load_or_build_index(self):
        if not os.path.exists(f"{FAISS_DIR}/index.faiss"):
            print("Building FAISS index...")
            docs = get_documents()
            vectorstore = FAISS.from_documents(docs, self.embedding_model)
            vectorstore.save_local(FAISS_DIR)
        else:
            print("Loading existing FAISS index...")
            vectorstore = FAISS.load_local(
                FAISS_DIR,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        return vectorstore

    def search(self, query):
        return self.retriever.invoke(query)