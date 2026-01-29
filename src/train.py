import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
import numpy as np
import json
import pickle
from dataset import NERDataset
from model import create_model

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/bert_ner")
BATCH_SIZE = 8
EPOCHS = 4
LR = 3e-5


def load_data():
    # Load encodings saved as pickle dictionaries
    with open(DATA_DIR / "X_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    with open(DATA_DIR / "X_val.pkl", "rb") as f:
        X_val = pickle.load(f)

    # Load label arrays
    y_train = np.load(DATA_DIR / "y_train.npy", allow_pickle=True)
    y_val = np.load(DATA_DIR / "y_val.npy", allow_pickle=True)

    return X_train, X_val, y_train, y_val


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Load data
    X_train, X_val, y_train, y_val = load_data()

    # Load label map
    with open(DATA_DIR / "label2id.json") as f:
        label2id = json.load(f)

    num_labels = len(label2id)

    # Build datasets
    train_dataset = NERDataset(X_train, y_train)
    val_dataset = NERDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Create model
    model = create_model(num_labels)
    model.to(device)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"📘 Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Model and tokenizer saved to models/bert_ner")


if __name__ == "__main__":
    train()
