# NHS Disease Information Chatbot (LLM + RAG)

A retrieval-augmented chatbot that answers questions about diseases, symptoms, and treatments **strictly using official NHS documents**.  
The application is designed to provide **grounded, source-based medical information**, avoiding general or unsupported responses.

🔗 **Live application:**  
https://nhsa2z.streamlit.app/

---

## Overview

This project demonstrates how Large Language Models (LLMs) can be combined with document retrieval to answer healthcare-related questions using **trusted NHS sources only**.

The chatbot:
- Retrieves relevant sections from NHS disease documents
- Generates answers based only on retrieved content
- Avoids answering non-medical or unrelated questions
- Provides NHS references where available

This is intended for **informational and educational purposes only**.

---

## How the System Works

### 1. Document Processing
- NHS disease PDFs are loaded and processed
- Text is split into manageable chunks
- Disease and source metadata is preserved

### 2. Vector Search
- Text chunks are converted into embeddings
- FAISS is used to retrieve the most relevant content for a query

### 3. Answer Generation
- Retrieved context is passed to the language model
- Responses are generated using only the provided NHS content
- If relevant information is not found, the model responds accordingly

### 4. User Interface
- Built using Streamlit
- Simple input-output interface for querying medical topics

---

## Key Features

- Retrieval-Augmented Generation (RAG)
- FAISS-based semantic search
- Context-restricted LLM responses
- Disease-specific document ingestion
- Streamlit deployment

---

## Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **OpenAI API**
- **FAISS**
- **PyMuPDF / PyPDF**
- **NLTK**

---

## Project Status

This repository represents a **functional prototype** demonstrating end-to-end RAG for medical information retrieval.

The deployed application shows:
- Working document ingestion
- Semantic retrieval using FAISS
- LLM-based answer generation
- Live Streamlit interface

A separate refactor focusing on production readiness, configuration management, and error handling is currently in progress and will be reflected in future updates.

---

## Disclaimer

This application provides **general medical information** derived from NHS documents.  
It does **not** offer medical advice, diagnosis, or treatment.

For personal medical concerns, users should consult a qualified healthcare professional.

---
