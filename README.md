# NHS Disease RAG System

A retrieval-based medical QA system that answers questions using NHS documents only.

This project focuses on one thing: giving answers that are **grounded, traceable, and safe**.  
If the system is not confident, it does not guess.

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
## Teh System:

- Accepts a user query about diseases, symptoms, or treatment
- Retrieves relevant NHS document sections
- Filters and validates the context
- Generates a response strictly from that context
- Verifies the response before showing it to the user

If any step fails, the system returns a safe fallback instead of forcing an answer.

---

## System flow

<!-- [Flowchart here] -->

RAG pipeline:

Query  
→ Classification  
→ Disease resolution  
→ Hybrid retrieval (BM25 + FAISS)  
→ Context validation  
→ LLM generation  
→ Grounding validation  
→ Response formatting  

---

## Retrieval layer

The system uses a hybrid approach:

- FAISS (dense retrieval) for semantic matching  
- BM25 (sparse retrieval) for keyword precision  

Results are merged before being passed forward.

Each chunk carries metadata:
- disease
- section (symptoms, treatment, etc.)
- source document

---

## Guardrails (before generation)

Before the model is called, the system checks:

- Do we have enough documents/chunks/retrived data?
- Are they from the same disease?
- Do they actually relate to the query?

If the context is weak, the system stops here.

This prevents the model from generating answers on weak ground.

---

## Grounding (after generation)

After the response is generated, it is validated again.

Checks include:
- Does the response overlap with the retrieved context?
- Does it mention the correct disease?
- Does it contain unsupported links?

If the answer fails validation, it is discarded.

---

## Failure handling

The system explicitly models failure states:

- Non-medical query
- Ambiguous symptoms
- Weak context
- Ungrounded response

Each case is handled with a safe, user-friendly message.

The system is designed to fail safely.

---

## Evaluation

The system is tested using a structured query set.

Metrics tracked:
- Disease prediction accuracy
- Refusal accuracy
- Answer rate

<!-- [Add results table here later] -->

Evaluation uses the same pipeline as the live system.

---

## Observability

Each query goes through multiple stages.

The system tracks:
- query type
- resolved disease
- retrieval behavior
- validation outcomes
- final decision path

This makes it possible to debug where and why the system failed.

---

## Latency

Planned:
- stage-wise latency tracking
- total response time monitoring

---

## Tech stack

- Python 3.12
- Streamlit  
- OpenAI API  
- LangChain  
- FAISS  
- BM25 (rank-bm25)  
- PyMuPDF / PyPDF  
- NLTK  

Refer requirements.txt for packages stable version.

---

## Disclaimer

This system provides general medical information based on NHS documents.

It is **not** a substitute for professional medical advice.  
Always consult a qualified healthcare provider for personal concerns.

---
<!-- ## How the System Works

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

--- -->
