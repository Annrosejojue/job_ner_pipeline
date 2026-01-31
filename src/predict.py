import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from pathlib import Path
import json

# Paths
MODEL_DIR = Path("models/bert_ner")
LABEL_MAP_PATH = Path("data/processed/label2id.json")

class NERPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # FIX: assign tokenizer to self
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
        # Load your fine‑tuned model
        self.model = BertForTokenClassification.from_pretrained(MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()

        # Load label map
        with open(LABEL_MAP_PATH) as f:
            label2id = json.load(f)
        self.id2label = {v: k for k, v in label2id.items()}

    def predict(self, text):
        """
        Correct inference tokenization:
        - Use raw text
        - Let BERT handle subword tokenization
        """

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            is_split_into_words=False
        )

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        # Convert IDs back to tokens + labels
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        labels = [self.id2label[i] for i in preds]

        # Remove special tokens
        clean_tokens, clean_labels = [], []
        for tok, lab in zip(tokens, labels):
            if tok not in ["[CLS]", "[SEP]", "[PAD]"]:
                clean_tokens.append(tok)
                clean_labels.append(lab)

        return clean_tokens, clean_labels
