# --------------------------------------------------------------------------------
# GUARDRAILS - context and response validation
# --------------------------------------------------------------------------------
# Validating retrieved documents before generation
# Preventing weak or irrelevant context from producing misleading answers
# --------------------------------------------------------------------------------

def is_valid_context(docs, query):
    if not docs:
        return False

    query_terms = set(query.lower().split())

    strong_match_count = 0

    for d in docs:
        doc_terms = set(d.page_content.lower().split())

        overlap = query_terms.intersection(doc_terms)

        # requiring stronger overlap
        if len(overlap) >= 3:
            strong_match_count += 1

    # requiring atleast 2 strong mathces
    if strong_match_count < 2:
        return False

    return True


def is_valid_response(response):
    if not response:
        return False

    if len(response.strip()) < 50:
        return False

    return True