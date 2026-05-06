# --------------------------------------------------------------------------------
# Response Formatting Layer
# Converting internal system flags into clean messages
# --------------------------------------------------------------------------------

class ResponseFormatter:
    def __init__(self):
        pass

    def format(self, response):
        if response == "[NON_MEDICAL]":
            return "This system is designed to answer NHS medical-related questions only."

        if response.startswith("[AMBIGUOUS_QUERY]"):
            return (
                "I'm not confident enough to map your symptoms to a specific condition from the available NHS data.\n\n"
                "Try asking with more detail or mention a specific condition if you have one in mind."
            )

        # Weak or insufficient retrieval context
        if response == "[CTX_FAIL]":
            return (
                "I couldn't find strong enough information in the NHS dataset to answer this reliably.\n\n"
                "This usually happens when the symptoms are too broad or don't clearly match one condition."
            )

        # Generated answer not grounded in retrieved content
        if response == "[GROUND_FAIL]":
            return (
                "I found some related information, but I'm not confident the answer would be fully supported by the NHS data.\n\n"
                "To avoid giving misleading information, I'm not providing an answer."
            )
        
        # Source mismatch or missing disease reference
        if response == "[SOURCE_FAIL]":
            return (
                "The generated answer didn't clearly match a specific NHS condition, so it has been withheld."
            )
        
        # LLM failed to produce a usable response
        if response == "[RESP_FAIL]":
            return "I wasn't able to generate a reliable answer from the available data."

        # Blocking hallucinated or external links
        if response == "[LINK_FAIL]":
            return (
                "The generated response included unsupported external references, so it has been blocked."
            )

        return response