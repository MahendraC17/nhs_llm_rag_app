# --------------------------------------------------------------------------------
# FAISS-based RAG pipeline (LCEL, LangChain 1.2.7 compatible)
# --------------------------------------------------------------------------------

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from data_chunking import load_and_chunk_pdfs
from config.settings import Settings

settings = Settings()


def get_embedding_model():
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key
    )


def build_faiss_index():
    embedding_model = get_embedding_model()
    docs = load_and_chunk_pdfs()

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(settings.faiss_dir)

    return vectorstore


def load_faiss_index():
    """
    Load existing FAISS index from disk.
    """
    embedding_model = get_embedding_model()

    return FAISS.load_local(
        settings.faiss_dir,
        get_embedding_model(),
        allow_dangerous_deserialization=True
    )

def get_retriever(k: int = 5):
    """
    Load FAISS index and return retriever.
    """
    if not os.path.exists(os.path.join(settings.faiss_dir, "index.faiss")):
        raise FileNotFoundError("FAISS index not found. Building it.")

    vectorstore = load_faiss_index()
    return vectorstore.as_retriever(search_kwargs={"k": k})

def format_docs(docs):
    formatted = []

    for doc in docs:
        disease = doc.metadata.get("disease", "Unknown")
        formatted.append(
            f"[Disease: {disease}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted)



def get_rag_chain():
    """
    Construct and return the RAG LCEL chain.
    """
    retriever = get_retriever()
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key
    )
    
    prompt = PromptTemplate(
        template=settings.prompt_template,
        input_variables=["context", "question"]
        )

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

#### TEsting
test_chain = get_rag_chain()
print(test_chain.invoke("What is autism?"))
