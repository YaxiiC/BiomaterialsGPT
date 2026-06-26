import config


def _context_clause(context):
    if config.ALLOW_CONTEXT_OPTIONAL:
        return (
            "You may refer to the context below if it helps. "
            "If the context is irrelevant or unhelpful, ignore it and answer "
            "from your own knowledge.\n\n"
            f"CONTEXT:\n{context}\n"
        )
    return f"Answer based on the context below.\n\nCONTEXT:\n{context}\n"


def _source_tag_clause():
    if config.TAG_ANSWER_SOURCE:
        return ("\nOn a new final line, add SOURCE: CONTEXT if you used the "
                "context, or SOURCE: OWN if you answered from your own knowledge.")
    return ""


def mcq(question, options, context):
    opts = "\n".join(f"{k}) {v}" for k, v in options.items())
    return (
        "You are a biomaterials expert answering a single-choice question.\n\n"
        f"{_context_clause(context)}\n"
        f"QUESTION: {question}\n\nOPTIONS:\n{opts}\n\n"
        "Respond with the answer as: ANSWER: <letter>"
        f"{_source_tag_clause()}"
    )


def open_ended(question, context):
    return (
        "You are a biomaterials expert. Answer clearly and factually in under "
        "200 words.\n\n"
        f"{_context_clause(context)}\n"
        f"QUESTION: {question}\n\nANSWER:"
        f"{_source_tag_clause()}"
    )
