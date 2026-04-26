from retrieval.bm25_engine import BM25Engine
from retrieval.faiss_engine import FAISSEngine

class HybridRetriever:
    def __init__(self, k_dense=5, k_sparse=5, final_k=5):
        self.k_dense = k_dense
        self.k_sparse = k_sparse
        self.final_k = final_k

        self.bm25 = BM25Engine()
        self.faiss = FAISSEngine()

    def search(self, query):
        
        dense_docs = self.faiss.search(query)
        sparse_docs = self.bm25.search(query, k=self.k_sparse)

        scored = {}

        for rank, doc in enumerate(dense_docs):
            key = (doc.page_content, doc.metadata.get("source", ""))
            score = 1 / (rank + 1) 
            scored[key] = scored.get(key, 0) + (0.7 * score)

        for rank, doc in enumerate(sparse_docs):
            key = (doc.page_content, doc.metadata.get("source", ""))
            score = 1 / (rank + 1)
            scored[key] = scored.get(key, 0) + (0.3 * score)

        # Sort by combined score
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        # Map back to docs
        content_to_doc = {
            doc.page_content: doc
            for doc in dense_docs + sparse_docs
        }

        final_docs = [content_to_doc[c] for c, _ in ranked[:self.final_k]]

        return final_docs