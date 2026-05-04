from collections import Counter


STOPWORDS = {
    "what", "how", "is", "the", "a", "an", "do", "does", "did",
    "i", "you", "we", "they", "it", "to", "of", "in", "on", "for",
    "with", "and", "or", "by", "this", "that", "are", "was", "were"
}


def is_definition_query(query):
    q = query.lower()
    return any(phrase in q for phrase in ["what is", "define", "explain"])


def extract_dominant_disease(docs):
    if not docs:
        return None

    diseases = [
        d.metadata.get("disease", "").lower()
        for d in docs if d.metadata
    ]

    if not diseases:
        return None

    return Counter(diseases).most_common(1)[0][0]


def filter_to_dominant_disease(docs):
    dominant = extract_dominant_disease(docs)

    if not dominant:
        return docs

    return [
        d for d in docs
        if d.metadata.get("disease", "").lower() == dominant
    ]


def build_context_text(docs):
    return " ".join(d.page_content.lower() for d in docs)


def extract_meaningful_tokens(text):
    tokens = text.lower().split()

    return [
        t for t in tokens
        if t not in STOPWORDS and len(t) > 3
    ]


def is_grounded_response(response, docs, query):
    if not response or not docs:
        return False

    context_text = build_context_text(docs)
    tokens = extract_meaningful_tokens(response)

    if not tokens:
        return False

    match_count = sum(1 for t in tokens if t in context_text)
    ratio = match_count / len(tokens)

    threshold = 0.2 if is_definition_query(query) else 0.1

    return ratio >= threshold


def has_external_links(response):
    if not response:
        return False

    r = response.lower()
    return "http://" in r or "https://" in r


def is_valid_source(response, docs):
    dominant = extract_dominant_disease(docs)

    if not dominant:
        return False

    response_lower = response.lower()

    return any(part in response_lower for part in dominant.split())