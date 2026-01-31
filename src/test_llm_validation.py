from predict import NERPredictor
from postprocessing.merge_entities import merge_entities
from llm_agent.validator import LLMValidator
from cross_validation.cross_validator import CrossValidator

text = "Python Java SQL"

ner = NERPredictor()
tokens, labels = ner.predict(text)

bert_entities = merge_entities(tokens, labels)

validator = LLMValidator()
cross = CrossValidator(validator)

final_entities = cross.cross_validate(text, bert_entities)

print("BERT entities:", bert_entities)
print("Final validated entities:", final_entities)
