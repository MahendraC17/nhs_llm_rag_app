from rank_bm25 import BM25Okapi

class BM25Engine:
    def __init__(self, docs):
        self.docs = docs
        self.texts = [doc.page_content for doc in docs]
        self.tokenized = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query, k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [self.docs[i] for i in top_idx]