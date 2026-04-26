# --------------------------------------------------------------------------------
# RAG SERVICE - Orchestrating retrieval andd generation
# --------------------------------------------------------------------------------

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from retrieval.hybrid_retriever import HybridRetriever
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
        context = self._format_docs(docs)

        response = (
            self.prompt
            | self.llm
            | StrOutputParser()
        ).invoke({
            "context": context,
            "question": user_query
        })

        return response