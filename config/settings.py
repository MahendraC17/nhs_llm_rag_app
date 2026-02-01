from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from pathlib import Path


class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    embedding_model: str = Field(
        default="text-embedding-3-small",
        env="EMBEDDING_MODEL"
    )

    llm_model: str = Field(
        default="gpt-4o-mini",
        env="LLM_MODEL"
    )

    base_dir: Path = Path(__file__).resolve().parent.parent

    data_dir: Path = Field(..., env="DATA_DIR")
    faiss_dir: Path = Field(..., env="FAISS_DIR")

    prompt_template: str = """
        You are a helpful assistant specialized in answering questions strictly based on NHS documents related to diseases, symptoms, and treatments.
        - If the question is unrelated to medical or NHS disease-related topics, respond with: "I don't know."
        - If the answer cannot be found in the provided context, respond with: "This platform provides medical information related to diseases, symptoms, and treatment based on NHS documents."

        - Do not add information from outside the context.

        Use only the retrieved context to answer the user's question.
        If relevant content is found:
        - Format your response as:
        - Concise bullet points
        - Include any self-care advice mentioned in the context
        - If the document mentions symptoms requiring urgent medical attention, highlight those using a [Red Flag] tag
        - The NHS disease link must follow this format :https://www.nhs.uk/conditions/<disease-name>/ where <disease-name> is lowercase and hyphenated.
        - Always end the response with:Source : (Name of the disease) https://www.nhs.uk/conditions/<disease-name>/


        Context:
        {context}

        Question:
        {question}

        Answer:

        """

    @field_validator("data_dir", mode="before")
    @classmethod
    def set_data_dir(cls, v, info):
        return v or info.data["base_dir"] / "data"

    @field_validator("faiss_dir", mode="before")
    @classmethod
    def set_faiss_dir(cls, v, info):
        return v or info.data["base_dir"] / "faiss_index"

    @field_validator("openai_api_key")
    @classmethod
    def api_key_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("OPENAI_API_KEY cannot be empty")
        return v

    # 🔴 THIS IS THE IMPORTANT PART
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"   # ignore raw ENV keys like OPENAI_API_KEY
    )
