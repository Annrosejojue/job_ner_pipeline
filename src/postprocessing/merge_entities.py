def merge_entities(tokens, labels):
    entities = []
    current_tokens = []

    for tok, lab in zip(tokens, labels):
        if lab == "B-SKILL":
            if current_tokens:
                entities.append(_clean(current_tokens))
                current_tokens = []
            current_tokens.append(tok)

        elif lab == "I-SKILL":
            current_tokens.append(tok)

        else:
            if current_tokens:
                entities.append(_clean(current_tokens))
                current_tokens = []

    if current_tokens:
        entities.append(_clean(current_tokens))

    return [{"text": e, "label": "SKILL"} for e in entities]


def _clean(token_list):
    """Fix WordPiece tokens and join with spaces."""
    cleaned = []
    for tok in token_list:
        if tok.startswith("##"):
            cleaned[-1] = cleaned[-1] + tok[2:]
        else:
            cleaned.append(tok)
    return " ".join(cleaned)
