# --------------------------------------------------------------------------------
# Disease Resolution Layer
# Mapping ambiguous symptom queries to a known disease within dataset
# --------------------------------------------------------------------------------

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

                Query: "wheezing and shortness of breath"
                Answer: asthma

                Query: "painful swollen joints"
                Answer: arthritis

                Query: "joint stiffness after waking up"
                Answer: arthritis

                Query: "child struggles to sit still in class"
                Answer: ADHD in children and young people

                Query: "low blood pressure and dizziness"
                Answer: Addison's disease

                Query: "heel pain first thing in the morning"
                Answer: plantar fasciitis

                Query:
                {query}

                Answer:
                """,
            input_variables=["query", "disease_list"])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def _load_diseases(self):
        # Extracting disease names from dataset files to avoid external mapping
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
        result = result.strip()
        if result.upper() == "NONE":
            return None

        result_clean = result.lower().rstrip(".")
        for disease in self.diseases:
            if disease.lower() == result_clean:
                return disease

        return None

    # --------------------------------------------------------------------------------
    # Main Resolver Entry Point
    # Running constrained mapping and returning dataset aligned disease or None
    # --------------------------------------------------------------------------------
    def match(self, query):
        result = self._invoke_llm(query)
        print(f"Resolver Query: {query}")
        print(f"Resolver Result: {result}")
        return self._validate_output(result)