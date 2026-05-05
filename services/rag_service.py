# --------------------------------------------------------------------------------
# RAG Orchestration Layer
# Coordinating classification, retrieval, validation, and generation into one flow
# --------------------------------------------------------------------------------

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
from services.logger import RAGLogger
from config import LLM_MODEL, TEMPLATE, OPENAI_API_KEY


class RAGService:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        # Initializing core components used across the pipeline
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
        # Flattening retrieved documents into a single context block
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _is_direct_disease_query(self, query):
        query_lower = query.lower()

        for disease in self.resolver.diseases:
            if disease.lower() in query_lower:
                return True

        return False

    # --------------------------------------------------------------------------------
    # Core Pipeline Execution
    # Applying retrieval filtering -> validation -> generation -> grounding
    # --------------------------------------------------------------------------------
    def _run_pipeline(self, user_query, docs, logger):
        # Narrowing context to dominant disease to avoid cross disease mixing
        docs = filter_to_dominant_disease(docs)

        # Stopping early if context is weak or inconsistent
        valid_ctx = is_valid_context(docs, user_query)
        logger.log_guardrail(valid_ctx)

        if not valid_ctx:
            return "[CTX_FAIL]"

        context = self._format_docs(docs)

        # Running generation on the above filtered context
        response = (
            self.prompt
            | self.llm
            | StrOutputParser()
        ).invoke({
            "context": context,
            "question": user_query
        })

        logger.log_generation(response)

        # Validating response quality and safety
        if not is_valid_response(response):
            return "[RESP_FAIL]"

        if has_external_links(response):
            return "[LINK_FAIL]"

        grounded = is_grounded_response(response, docs, user_query)
        logger.log_grounding(grounded)

        if not grounded:
            return "[GROUND_FAIL]"

        if not is_valid_source(response, docs):
            return "[SOURCE_FAIL]"

        return response

    # --------------------------------------------------------------------------------
    # Query Routing Entry Point
    # Handling query classification, disease resolution, and retrieval strategy
    # Ensuring consistent execution path for both normal and evaluation flows
    # --------------------------------------------------------------------------------
    def _execute_query(self, user_query, docs=None):
        logger = RAGLogger()
        logger.start(user_query)

        if docs is None:
            # Skipping classifier if disease clearly present
            if self._is_direct_disease_query(user_query):
                query_type = "KNOWN_DISEASE"
            else:
                query_type = self.classifier.classify(user_query)

            logger.log_query_type(query_type)

            if query_type == "NON_MEDICAL":
                logger.set_final_status("NON_MEDICAL")
                logger.end()
                return self.formatter.format("[NON_MEDICAL]")

            if query_type == "AMBIGUOUS_MEDICAL":
                disease = self.resolver.match(user_query)
                logger.log_resolved_disease(disease)

                if disease:
                    docs = self.retriever.search(disease)
                else:
                    logger.set_final_status("AMBIGUOUS_QUERY")
                    logger.end()
                    return self.formatter.format("[AMBIGUOUS_QUERY]")

            else:
                docs = self.retriever.search(user_query)

        # log retrieval
        logger.log_retrieval(docs)

        result = self._run_pipeline(user_query, docs, logger)

        logger.set_final_status(result if result.startswith("[") else "SUCCESS")
        logger.end()

        return self.formatter.format(result)

    def query(self, user_query):
        return self._execute_query(user_query)

    def query_with_docs(self, user_query, docs):
        return self._execute_query(user_query, docs)