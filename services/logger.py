# --------------------------------------------------------------------------------
# Logging Layer
# Building structured log per query without affecting pipeline behavior
# --------------------------------------------------------------------------------

import json
import time
from collections import Counter


class RAGLogger:
    def __init__(self):
        self.log = {}
        self.start_time = None

    def start(self, query):
        self.start_time = time.time()
        self.log = {
            "query": query,
            "query_type": None,
            "resolved_disease": None,
            "retrieved_docs_count": 0,
            "disease_distribution": {},
            "guardrail_passed": None,
            "response_generated": False,
            "grounding_passed": None,
            "final_status": None,
            "latency_ms": None
        }

    def log_query_type(self, query_type):
        self.log["query_type"] = query_type

    def log_resolved_disease(self, disease):
        self.log["resolved_disease"] = disease

    def log_retrieval(self, docs):
        self.log["retrieved_docs_count"] = len(docs)

        diseases = [
            d.metadata.get("disease", "").lower()
            for d in docs if d.metadata
        ]

        if diseases:
            self.log["disease_distribution"] = dict(Counter(diseases))

    def log_guardrail(self, passed):
        self.log["guardrail_passed"] = passed

    def log_generation(self, response):
        self.log["response_generated"] = bool(response and response.strip())

    def log_grounding(self, passed):
        self.log["grounding_passed"] = passed

    def set_final_status(self, status):
        self.log["final_status"] = status

    def end(self):
        self.log["latency_ms"] = int((time.time() - self.start_time) * 1000)

        try:
            print(json.dumps(self.log, indent=2))
        except Exception:
            pass