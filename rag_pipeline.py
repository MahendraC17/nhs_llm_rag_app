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
from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    FAISS_DIR,
    TEMPLATE
)

def get_embedding_model():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )


def build_faiss_index():
    embedding_model = get_embedding_model()
    docs = load_and_chunk_pdfs()

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(FAISS_DIR)

    return vectorstore


def load_faiss_index():
    """
    Load existing FAISS index from disk.
    """
    embedding_model = get_embedding_model()

    return FAISS.load_local(
        FAISS_DIR,
        embedding_model,
        allow_dangerous_deserialization=True
    )

def get_retriever(k: int = 5):
    """
    Load FAISS index and return retriever.
    """
    if not os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
        raise FileNotFoundError("FAISS index not found. Building it.")

    vectorstore = load_faiss_index()
    return vectorstore.as_retriever(search_kwargs={"k": k})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    """
    Construct and return the RAG LCEL chain.
    """
    retriever = get_retriever()
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY
        )
    
    prompt = PromptTemplate(
        template=TEMPLATE,
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
