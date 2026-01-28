import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report
import numpy as np
import json
from pathlib import Path
from dataset import NERDataset
from model import create_model

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/bert_ner")

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_val = np.load(DATA_DIR / "X_val.npy", allow_pickle=True).item()
    y_val = np.load(DATA_DIR / "y_val.npy", allow_pickle=True)

    with open(DATA_DIR / "label2id.json") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    val_dataset = NERDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = create_model(len(label2id))
    model.load_state_dict(torch.load(MODEL_DIR / "pytorch_model.bin"))
    model.to(device)
    model.eval()

    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            for p, l in zip(preds, labels):
                p = p.cpu().numpy().tolist()
                l = l.cpu().numpy().tolist()

                true_seq, pred_seq = [], []
                for pi, li in zip(p, l):
                    if li == -100:
                        continue
                    true_seq.append(id2label[li])
                    pred_seq.append(id2label[pi])

                all_true.append(true_seq)
                all_preds.append(pred_seq)

    print(classification_report(all_true, all_preds))

if __name__ == "__main__":
    evaluate()
