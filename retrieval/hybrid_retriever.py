# --------------------------------------------------------------------------------
# Hybrid Retrieval Layer
# Combining dense (FAISS) and sparse (BM25) results with weighted scoring
# --------------------------------------------------------------------------------

from retrieval.bm25_engine import BM25Engine
from retrieval.faiss_engine import FAISSEngine


class HybridRetriever:
    def __init__(self, k_dense=5, k_sparse=5, final_k=5):
        self.k_dense = k_dense
        self.k_sparse = k_sparse
        self.final_k = final_k

        self.bm25 = BM25Engine()
        self.faiss = FAISSEngine()

    # --------------------------------------------------------------------------------
    # Hybrid Search Entry Point
    # Merging results from both retrievers
    #
    # Dense retrieval captures semantic meaning
    # Sparse retrieval reinforces keyword precision
    # --------------------------------------------------------------------------------
    def search(self, query):

        dense_docs = self.faiss.search(query)
        sparse_docs = self.bm25.search(query, k=self.k_sparse)

        scored = {}

        # Weighting dense results higher for semantic relevance
        for rank, doc in enumerate(dense_docs):
            key = doc.page_content
            score = 1 / (rank + 1)
            scored[key] = scored.get(key, 0) + (0.7 * score)

        # Adding sparse scores for lexical alignment
        for rank, doc in enumerate(sparse_docs):
            key = doc.page_content
            score = 1 / (rank + 1)
            scored[key] = scored.get(key, 0) + (0.3 * score)

        # Sorting combined results
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        # Mapping content back to document objects
        content_to_doc = {
            doc.page_content: doc
            for doc in dense_docs + sparse_docs
        }

        final_docs = [content_to_doc[c] for c, _ in ranked[:self.final_k]]

        return final_docs