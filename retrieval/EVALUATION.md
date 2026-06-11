# Evaluation Report

## Overview

Evaluation of the output was measured using a manually curated test set containing 100 queries covering:

* Direct disease queries
* Symptom-based medical queries
* Ambiguous medical queries
* Non-medical queries
* Out-of-scope queries

The goal was to measure the effectiveness of the complete RAG pipeline.

The evaluation process used the full production workflow:

```
Query
-> Query Classification
-> Disease Resolution
-> Hybrid Retrieval
-> Context Validation
-> LLM Generation
-> Grounding Validation
-> Final Response
```

---

# Evaluation Categories

## Normal Queries

Queries that should successfully retrieve relevant NHS information with the documents provided and generate a grounded response.

Success criteria:

* Query successfully reaches retrieval
* Valid context is found
* Grounded response is generated
* Final status is SUCCESS

---

## Ambiguous Medical Queries

Queries describing symptoms rather than explicitly naming a disease or queries describing out of scope from the uploaded mediacal PDFs.

Success criteria:

* DiseaseResolver correctly maps symptoms to a disease

or

* System safely returns AMBIGUOUS_QUERY when confidence is insufficient

---

## Refusal Queries

Queries outside the medical scope of the NHS dataset and diseases in general.


Success criteria:

* Query classified as NON_MEDICAL
* System refuses to answer
* Final status is NON_MEDICAL

---

# Evaluation Methodology

The evaluator runs each query through the same production pipeline used by end users.

For each query, the following information is recorded:

* Query
* Expected category
* Actual query type
* Resolved disease
* Response
* Final status

The evaluator then computes:

1. Normal Query Accuracy
2. Refusal Accuracy
3. Ambiguous Query Accuracy
4. Successful Answer Count

---

# Final Results

Dataset Size: 100 Queries (evaluation/test_set.csv)

| Metric                   | Score |
| ------------------------ | ----- |
| Normal Query Accuracy    | 92%   |
| Refusal Accuracy         | 96%   |
| Ambiguous Query Accuracy | 80%   |
| Successful Answers       | 65    |

---

# Key Improvements During Evaluation

## Hybrid Retrieval

Implemented a hybrid retrieval strategy combining:

* FAISS dense semantic search
* BM25 keyword retrieval

This improved robustness across both disease-specific and symptom-based queries.

---

## Full Pipeline Evaluation

The original evaluator measured retrieval output only.

The evaluation framework was redesigned to exercise:

* Query Classification
* Disease Resolution
* Retrieval
* Guardrails
* Grounding
* Response Generation

---

## Disease Resolver Validation Fix

A significant issue was discovered in the disease resolution layer.

The resolver was correctly identifying diseases, but valid outputs were rejected due to strict string matching.

Example:

```
Resolver Result: asthma
Resolved Disease: None
```

After implementing case-insensitive validation and normalization, the resolver successfully accepted valid disease matches.

This change improved Normal Query Accuracy from approximately 77% to 92%.

---

# Failure Analysis

Most remaining failures occur during disease resolution rather than retrieval.

Examples include:

* fatigue and pale skin
* sudden abdominal pain with nausea
* child constantly interrupting others
* ongoing fatigue with low blood pressure

These failures occur because symptom descriptions are not mapped confidently to a single disease in the dataset.

The system intentionally favors conservative behavior and returns AMBIGUOUS_QUERY rather than risking incorrect medical guidance.

---

# Limitations

Current limitations include:

* Dependence on an LLM for symptom-to-disease resolution
* Closed-dataset design limited to NHS documents
* No multi-hop reasoning across diseases
* No support for user-uploaded documents
* Ambiguous symptom descriptions may remain unresolved

---