import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import DATA_DIR, LLM_MODEL, OPENAI_API_KEY


class DiseaseResolver:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        self.diseases = self._load_diseases()
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

        self.prompt = PromptTemplate(
            template="""
                You are mapping a user query to a known NHS disease dataset.

                Available diseases:
                {disease_list}

                Task:
                - Choose the MOST relevant disease ONLY if clearly supported by the query
                - If the query strongly suggests ONE disease, return it
                - If multiple diseases could match OR signal is weak, return NONE
                - If no disease clearly matches, return NONE
                - Do NOT invent new diseases
                - Only return EXACT disease name from the list OR NONE

                Examples:

                Query: "persistent cough and shortness of breath"
                Answer: asthma

                Query: "my child can't focus and is restless"
                Answer: ADHD in children and young people

                Query: "yellowing of skin and tiredness"
                Answer: NONE

                Query:
                {query}

                Answer:
                """,
            input_variables=["query", "disease_list"]
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def _load_diseases(self):
        return [
            f.replace("_", " ").replace(".pdf", "")
            for f in os.listdir(DATA_DIR)
            if f.endswith(".pdf")
        ]

    def _invoke_llm(self, query):
        return self.chain.invoke({
            "query": query,
            "disease_list": ", ".join(self.diseases)
        }).strip()

    def _validate_output(self, result):
        if result == "NONE":
            return None

        if result in self.diseases:
            return result

        return None

    def match(self, query):
        result = self._invoke_llm(query)
        return self._validate_output(result)