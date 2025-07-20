import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from collections import defaultdict
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from data_chunking import load_and_chunk_pdfs
from config import OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL, FAISS_DIR

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

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

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0
)

from langchain.chains import RetrievalQAWithSourcesChain

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "What causes acute pancreatis? only give cause and no discription include minor causes as well"
response = qa_chain.invoke({"question": query})
print(response["answer"])
print("Sources:", response["sources"])