class CrossValidator:
    def __init__(self, llm_validator):
        self.llm = llm_validator

    def cross_validate(self, text, bert_entities):
        # LLM now returns a LIST of final entities
        llm_entities = self.llm.validate(text, bert_entities)

        # If LLM returns nothing, fallback to BERT
        if not llm_entities:
            return bert_entities

        # LLM output is already the final merged list
        return llm_entities
