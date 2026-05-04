# --------------------------------------------------------------------------------
# GUARDRAILS - context and response validation
# --------------------------------------------------------------------------------
# Validating retrieved documents before generation
# Preventing weak or irrelevant context from producing misleading answers
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
    for d in docs:
        doc_tokens = set(d.page_content.lower().split())
        if query_tokens.intersection(doc_tokens):
            return True
    return False


def is_valid_context(docs, query):
    if not docs:
        return False

    diseases = _extract_diseases(docs)

    if not diseases:
        return False

    _, dominant_count = _get_dominant_disease_count(diseases)

    if dominant_count < 2:
        return False

    if dominant_count >= 3:
        return True

    query_tokens = _extract_query_tokens(query)

    if not query_tokens:
        return False

    return _has_query_overlap(query_tokens, docs)


def is_valid_response(response):
    if not response:
        return False

    if len(response.strip()) < 50:
        return False

    return True