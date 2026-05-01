# --------------------------------------------------------------------------------
# RAG SERVICE - Orchestrating retrieval andd generation
# --------------------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from retrieval.hybrid_retriever import HybridRetriever
from services.guardrails import is_valid_context, is_valid_response
from services.grounding import filter_to_dominant_disease, is_grounded_response, has_external_links, is_valid_source
from config import LLM_MODEL, TEMPLATE, OPENAI_API_KEY


class RAGService:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        self.retriever = HybridRetriever()
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

        self.prompt = PromptTemplate(
            template=TEMPLATE,
            input_variables=["context", "question"]
        )

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, user_query):
        docs = self.retriever.search(user_query)

        # enforce single-disease context
        docs = filter_to_dominant_disease(docs)

        # --------------------------------------------------------------------------------
        # Context validation - guarddrail
        # --------------------------------------------------------------------------------

        if not is_valid_context(docs, user_query):
            return "I don't have enough relevant NHS information to answer that safely."

        context = self._format_docs(docs)

        response = (
            self.prompt
            | self.llm
            | StrOutputParser()
        ).invoke({
            "context": context,
            "question": user_query
        })

        # --------------------------------------------------------------------------------
        # Response validation - guarddrail
        # --------------------------------------------------------------------------------

        # response check
        if not is_valid_response(response):
            return "I'm not confident enough to provide a reliable NHS provided answer for that."

        # blocking external links
        if has_external_links(response):
            return "I don't have enough relevant NHS information to answer that safely."

        # enforcing grounding
        if not is_grounded_response(response, docs):
            return "I don't have enough relevant NHS information to answer that safely."

        # enforcing source correctness
        if not is_valid_source(response, docs):
            return "I don't have enough relevant NHS information to answer that safely."

        return response