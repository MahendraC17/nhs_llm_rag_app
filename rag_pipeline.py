# --------------------------------------------------------------------------------
# Loads or builds the FAISS index from PDF chunks using OpenAI embeddings
# Initializes the language model and sets up the RAG chain (chat_chain)
# Using the chat_chain for streamlit application or can be used directly 
# --------------------------------------------------------------------------------

import os
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from data_chunking import load_and_chunk_pdfs
from config import OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL, FAISS_DIR, TEMPLATE

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# --------------------------------------------------------------------------------
# FAISS INDEX SETUP (Semantic based)
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

# --------------------------------------------------------------------------------
# BM25 SETUP (Keyword based)
# --------------------------------------------------------------------------------
bm25_cache_path = os.path.join(FAISS_DIR, "bm25_cache.pkl")
if os.path.exists(bm25_cache_path):
    print("Loading BM25 retriever from cache...")
    with open(bm25_cache_path, "rb") as f:
        bm25_retriever = pickle.load(f)
else:
    print("Building BM25 retriever...")
    all_chunks = load_and_chunk_pdfs()
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 4
    with open(bm25_cache_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

# --------------------------------------------------------------------------------
# Ensemble Retriever: FAISS + BM25
# --------------------------------------------------------------------------------
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0
)

# Created a template to reject any unrelated queries
prompt = PromptTemplate(template=TEMPLATE, input_variables=["summaries", "question"])

# --------------------------------------------------------------------------------
# RAG Chain
# --------------------------------------------------------------------------------
chat_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# query = "What kind of scans I need to do for acute pancreatis"
# response = chat_chain.invoke({"question": query})
# print(response["answer"])
# print("Sources:", response["sources"])