from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from retrieval.hybrid_retriever import HybridRetriever
from services.guardrails import is_valid_context, is_valid_response
from services.grounding import (
    filter_to_dominant_disease,
    is_grounded_response,
    has_external_links,
    is_valid_source
)
from services.query_classifier import QueryClassifier
from services.disease_resolver import DiseaseResolver
from services.response_formatter import ResponseFormatter
from config import LLM_MODEL, TEMPLATE, OPENAI_API_KEY


class RAGService:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        self.retriever = HybridRetriever()
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

        self.classifier = QueryClassifier()
        self.resolver = DiseaseResolver()
        self.formatter = ResponseFormatter()

        self.prompt = PromptTemplate(
            template=TEMPLATE,
            input_variables=["context", "question"]
        )

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _run_pipeline(self, user_query, docs):
        docs = filter_to_dominant_disease(docs)

        if not is_valid_context(docs, user_query):
            return "[CTX_FAIL]"

        context = self._format_docs(docs)

        response = (
            self.prompt
            | self.llm
            | StrOutputParser()
        ).invoke({
            "context": context,
            "question": user_query
        })

        if not is_valid_response(response):
            return "[RESP_FAIL]"

        if has_external_links(response):
            return "[LINK_FAIL]"

        if not is_grounded_response(response, docs, user_query):
            return "[GROUND_FAIL]"

        if not is_valid_source(response, docs):
            return "[SOURCE_FAIL]"

        return response

    def _execute_query(self, user_query, docs=None):
        if docs is None:
            query_type = self.classifier.classify(user_query)

            if query_type == "NON_MEDICAL":
                return self.formatter.format("[NON_MEDICAL]")

            if query_type == "AMBIGUOUS_MEDICAL":
                disease = self.resolver.match(user_query)

                if disease:
                    docs = self.retriever.search(disease)
                else:
                    return self.formatter.format("[AMBIGUOUS_QUERY]")

            else:
                docs = self.retriever.search(user_query)

        result = self._run_pipeline(user_query, docs)
        return self.formatter.format(result)

    def query(self, user_query):
        return self._execute_query(user_query)

    def query_with_docs(self, user_query, docs):
        return self._execute_query(user_query, docs)