import csv
import os
from collections import Counter

from services.rag_service import RAGService
from retrieval.hybrid_retriever import HybridRetriever


class Evaluator:
    def __init__(self):
        self.rag = RAGService()
        self.retriever = self.rag.retriever

        self.input_path = os.path.join("evaluation", "test_set.csv")
        self.output_path = os.path.join("evaluation", "test_results.csv")

    def _is_refusal(self, response):
        response_lower = response.lower()

        if "don't have enough relevant nhs information" in response_lower:
            return True

        if "not confident enough" in response_lower:
            return True

        if response.strip().lower() in ["i don't know.", "i don't know"]:
            return True

        return False

    def run(self):
        results = []

        with open(self.input_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                query = row["query"]
                expected_disease = row.get("expected_disease", "")
                query_type = row.get("type", "")

                # -----------------------------
                # Single retrieval (source of truth)
                # -----------------------------
                docs = self.retriever.search(query)

                # predicted disease from SAME docs
                diseases = [
                    d.metadata.get("disease", "").lower()
                    for d in docs
                    if d.metadata
                ]

                if diseases:
                    predicted_disease = Counter(diseases).most_common(1)[0][0]
                else:
                    predicted_disease = ""

                # generate response using SAME docs
                response = self.rag.query_with_docs(query, docs)

                # detect refusal
                is_refusal = self._is_refusal(response)

                results.append({
                    "query": query,
                    "expected_disease": expected_disease,
                    "type": query_type,
                    "response": response,
                    "response_length": len(response),
                    "is_refusal": is_refusal,
                    "predicted_disease": predicted_disease
                })

        with open(self.output_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print("Evaluation run completed. Results saved to test_results.csv")