# -----------------------------------------
# Filter for keeping only dominant disease
# -----------------------------------------
def filter_to_dominant_disease(docs):
    if not docs:
        return docs

    disease_counts = {}

    for d in docs:
        disease = d.metadata.get("disease", "").lower()
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

    dominant = max(disease_counts, key=disease_counts.get)

    filtered = []
    for d in docs:
        if d.metadata.get("disease", "").lower() == dominant:
            filtered.append(d)

    return filtered


def is_grounded_response(response, docs):
    if not response or not docs:
        return False

    context_text = ""
    for d in docs:
        context_text += " " + d.page_content.lower()

    response_tokens = response.lower().split()

    STOPWORDS = {
        "what", "how", "is", "the", "a", "an", "do", "does", "did",
        "i", "you", "we", "they", "it", "to", "of", "in", "on", "for",
        "with", "and", "or", "by", "this", "that", "are", "was", "were"
    }

    meaningful_tokens = []
    for t in response_tokens:
        if t not in STOPWORDS and len(t) > 3:
            meaningful_tokens.append(t)

    if not meaningful_tokens:
        return False

    match_count = 0
    for t in meaningful_tokens:
        if t in context_text:
            match_count += 1

    ratio = match_count / len(meaningful_tokens)

    return ratio >= 0.3


def has_external_links(response):
    if not response:
        return False

    r = response.lower()

    if "http://" in r or "https://" in r:
        return True

    return False

def is_valid_source(response, docs):
    if not response or not docs:
        return False

    disease_counts = {}

    for d in docs:
        disease = d.metadata.get("disease", "").lower()
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

    dominant = max(disease_counts, key=disease_counts.get)

    if dominant in response.lower():
        return True

    return False