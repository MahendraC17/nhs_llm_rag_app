# --------------------------------------------------------------------------------
# Sparse Retrieval Layer (BM25)
# Ranking documents based on keyword overlap and term frequency
# --------------------------------------------------------------------------------

import re
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from ingestion.data_loader import get_documents


class BM25Engine:
    def __init__(self):
        self.docs = get_documents()
        self.texts = [doc.page_content for doc in self.docs]

        # Preprocessing text before building BM25 corpus
        self.tokenized = [self._clean_and_tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def _clean_and_tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = word_tokenize(text)
        return tokens

    # --------------------------------------------------------------------------------
    # BM25 Search Entry Point
    # Scoring all documents and selecting top-k based on keyword relevance
    # --------------------------------------------------------------------------------
    def search(self, query, k=5):
        tokenized_query = self._clean_and_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [self.docs[i] for i in top_idx]