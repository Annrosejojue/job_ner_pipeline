from transformers import BertForTokenClassification

def create_model(num_labels):
    return BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=num_labels
    )
