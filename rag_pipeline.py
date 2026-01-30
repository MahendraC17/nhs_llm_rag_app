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
from config import OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL, FAISS_DIR, TEMPLATE

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# --------------------------------------------------------------------------------
# FAISS SETUP
# --------------------------------------------------------------------------------
if not os.path.exists(f"{FAISS_DIR}/index.faiss"):
    print("Building FAISS index...")
    all_chunks = load_and_chunk_pdfs()
    vectorstore = FAISS.from_documents(all_chunks, embedding_model)
    vectorstore.save_local(FAISS_DIR)
else:
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embedding_model,
        allow_dangerous_deserialization=True
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --------------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------------
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# --------------------------------------------------------------------------------
# PROMPT
# --------------------------------------------------------------------------------
prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=["context", "question"]
)

# --------------------------------------------------------------------------------
# LCEL RAG PIPELINE
# --------------------------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

chat_chain = rag_chain

# --------------------------------------------------------------------------------
# LOCAL TEST
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print(chat_chain.invoke("What is autism?"))
