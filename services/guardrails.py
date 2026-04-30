# --------------------------------------------------------------------------------
# GUARDRAILS - context and response validation
# --------------------------------------------------------------------------------
# Validating retrieved documents before generation
# Preventing weak or irrelevant context from producing misleading answers
# --------------------------------------------------------------------------------

def is_valid_context(docs, query):
    if not docs:
        return False


    diseases = [d.metadata.get("disease", "").lower() for d in docs if d.metadata]

    if not diseases:
        return False

    disease_counts = {}
    for d in diseases:
        disease_counts[d] = disease_counts.get(d, 0) + 1

    dominant_disease = max(disease_counts, key=disease_counts.get)
    dominant_count = disease_counts[dominant_disease]

    # Here it should have at least 2 docs from same disease
    if dominant_count < 2:
        return False

    if dominant_count >= 3:
        return True

    STOPWORDS = {
        "what", "how", "is", "the", "a", "an", "do", "does", "did",
        "i", "you", "we", "they", "it", "to", "of", "in", "on", "for",
        "with", "and", "or", "by", "this", "that", "are", "was", "were"
    }

    query_tokens = set(
        word for word in query.lower().split()
        if word not in STOPWORDS and len(word) > 2
    )

    if not query_tokens:
        return False

    for d in docs:
        doc_tokens = set(d.page_content.lower().split())
        if query_tokens.intersection(doc_tokens):
            return True

    return False


def is_valid_response(response):
    if not response:
        return False

    if len(response.strip()) < 50:
        return False

    return True