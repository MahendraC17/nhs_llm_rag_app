# --------------------------------------------------------------------------------
# Loads or builds the FAISS index from PDF chunks using OpenAI embeddings
# Initializes the language model and sets up the RAG chain (chat_chain)
# Using the chat_chain for streamlit application or can be used directly 
# --------------------------------------------------------------------------------

import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from data_chunking import load_and_chunk_pdfs
from config import OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL, FAISS_DIR, TEMPLATE
from config import OPENAI_API_KEY

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

# Created a template to reject any unrelated queries
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chat_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# query = "What causes acute pancreatis? only give cause and no discription include minor causes as well"
# response = chat_chain.invoke({"question": query})
# print(response["answer"])
# print("Sources:", response["sources"])