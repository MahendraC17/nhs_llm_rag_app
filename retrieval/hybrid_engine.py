from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config.settings import Settings

settings = Settings()
from data_chunking import load_and_chunk_pdfs
from retrieval.bm25_engine import BM25Engine

class HybridRetriever:
    def __init__(self, k_dense=5, k_sparse=5):
        self.k_dense = k_dense
        self.k_sparse = k_sparse

        self.docs = load_and_chunk_pdfs()

        # BM25
        self.bm25 = BM25Engine(self.docs)

        # FAISS
        embedding_model = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key
                    )

        self.vectorstore = FAISS.load_local(
        settings.faiss_dir,
        embedding_model,
        allow_dangerous_deserialization=True
                    )
        self.faiss = self.vectorstore.as_retriever(
            search_kwargs={"k": k_dense}
        )

    def search(self, query):
        dense_docs = self.faiss.invoke(query)
        sparse_docs = self.bm25.search(query, k=self.k_sparse)

        combined = {}
        for doc in dense_docs + sparse_docs:
            combined[doc.page_content] = doc

        return list(combined.values())[:10]