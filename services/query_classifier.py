# --------------------------------------------------------------------------------
# Query Classification Layer
# Classifying incoming queries into intent categories before retrieval
# --------------------------------------------------------------------------------

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import LLM_MODEL, OPENAI_API_KEY


class QueryClassifier:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _build_prompt(self):
        # Training output to a closed set to avoid unpredictable labels
        return PromptTemplate(
            template="""
                You are a strict classifier for a medical RAG system.

                Classify the user query into ONE of these categories:

                1. KNOWN_DISEASE
                → Query clearly refers to a known disease or condition
                → Example: "symptoms of asthma", "what is arthritis"

                2. AMBIGUOUS_MEDICAL
                → Mentions symptoms or vague health concerns
                → No clear disease
                → Example: "yellowing of skin and tiredness", "pain in heel"

                3. NON_MEDICAL
                → Not related to diseases or NHS medical domain
                → Example: "tell me a joke", "how to invest money"

                Respond with ONLY ONE WORD:
                KNOWN_DISEASE or AMBIGUOUS_MEDICAL or NON_MEDICAL

                Query:
                {query}

                Answer:
                """,
            input_variables=["query"]
        )

    def _invoke_llm(self, query):
        return self.chain.invoke({"query": query}).strip().upper()

    def _validate_output(self, result):
        # Forcing output into known labels to prevent pipeline breakage
        valid_labels = ["KNOWN_DISEASE", "AMBIGUOUS_MEDICAL", "NON_MEDICAL"]

        if result not in valid_labels:
            return "AMBIGUOUS_MEDICAL"

        return result

    # --------------------------------------------------------------------------------
    # Running LLM classification and normalizing output for downstream routing
    # --------------------------------------------------------------------------------
    def classify(self, query):
        result = self._invoke_llm(query)
        return self._validate_output(result)