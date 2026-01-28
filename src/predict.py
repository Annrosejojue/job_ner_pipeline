import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from pathlib import Path
import json

MODEL_DIR = Path("models/bert_ner")
LABEL_MAP_PATH = Path("data/processed/label2id.json")

class NERPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
        self.model = BertForTokenClassification.from_pretrained(MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()

        with open(LABEL_MAP_PATH) as f:
            label2id = json.load(f)
        self.id2label = {v: k for k, v in label2id.items()}

    def predict(self, text):
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)[0].cpu().numpy()
            probs = probs[0].cpu().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        labels = [self.id2label[i] for i in preds]

        return tokens, labels, probs
