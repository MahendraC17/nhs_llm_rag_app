class ResponseFormatter:
    def __init__(self):
        pass

    def format(self, response):
        if response == "[NON_MEDICAL]":
            return "This system only answers questions related to NHS medical conditions."

        if response.startswith("[AMBIGUOUS_QUERY]"):
            return "Please provide more specific symptoms or mention a condition."

        if response == "[CTX_FAIL]":
            return "I don’t have enough relevant information to answer this safely."

        if response == "[GROUND_FAIL]":
            return "I’m not confident enough in the available information to give a reliable answer."

        if response == "[SOURCE_FAIL]":
            return "The available information is insufficient to confidently answer this question."
        
        if response == "[RESP_FAIL]":
            return "I couldn’t generate a reliable answer from the available information."

        if response == "[LINK_FAIL]":
            return "The response contained unsupported external references and was blocked."

        return response