# --------------------------------------------------------------------------------
# Guardrails Layer
# Validating retrieved context before generation to avoid weak or misleading answers
# --------------------------------------------------------------------------------

from collections import Counter


STOPWORDS = {
    "what", "how", "is", "the", "a", "an", "do", "does", "did",
    "i", "you", "we", "they", "it", "to", "of", "in", "on", "for",
    "with", "and", "or", "by", "this", "that", "are", "was", "were"
}


def _extract_diseases(docs):
    return [
        d.metadata.get("disease", "").lower()
        for d in docs if d.metadata
    ]


def _get_dominant_disease_count(diseases):
    if not diseases:
        return None, 0

    counter = Counter(diseases)
    dominant, count = counter.most_common(1)[0]
    return dominant, count


def _extract_query_tokens(query):
    return {
        word for word in query.lower().split()
        if word not in STOPWORDS and len(word) > 2
    }


def _has_query_overlap(query_tokens, docs):
    # Checking whether at least one retrieved chunk shares meaningful tokens with query
    for d in docs:
        doc_tokens = set(d.page_content.lower().split())
        if query_tokens.intersection(doc_tokens):
            return True
    return False


# --------------------------------------------------------------------------------
# Context Validation Entry Point
# Ensuring retrieved documents are strong enough before passing to LLM
#
# Logic:
# - Require minimum signal from same disease
# - If strong dominance (>=3), allow directly
# - If borderline (2), require query overlap check
# --------------------------------------------------------------------------------
def is_valid_context(docs, query):
    if not docs:
        return False

    diseases = _extract_diseases(docs)

    if not diseases:
        return False

    _, dominant_count = _get_dominant_disease_count(diseases)

    # Weak signal, high risk of irrelevant context
    if dominant_count < 2:
        return False

    # Strong signal, safe to proceed
    if dominant_count >= 3:
        return True

    # Borderline case -> enforcing query relevance
    query_tokens = _extract_query_tokens(query)

    if not query_tokens:
        return False

    return _has_query_overlap(query_tokens, docs)


# --------------------------------------------------------------------------------
# Response Validation
# Filtering out extremely short or empty outputs before grounding checks
# --------------------------------------------------------------------------------
def is_valid_response(response):
    if not response:
        return False

    if len(response.strip()) < 50:
        return False

    return True