import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
import json

DATA_DIR = Path("data/processed")
IOB_PATH = DATA_DIR / "corpus_iob.csv"
LABEL_MAP_PATH = DATA_DIR / "label2id.json"
MAX_LEN = 128

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

def encode_tags(tags_list, encodings, label2id):
    encoded_labels = []
    for i, tags in enumerate(tags_list):
        word_ids = encodings.word_ids(batch_index=i)
        labels = []
        tag_seq = tags.split()

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            else:
                labels.append(label2id.get(tag_seq[word_id], label2id["O"]))
        encoded_labels.append(labels)
    return encoded_labels

def preprocess():
    df = pd.read_csv(IOB_PATH)

    # Build label map
    all_tags = sorted(list({tag for seq in df["tags"] for tag in seq.split()}))
    label2id = {tag: i for i, tag in enumerate(all_tags)}

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label2id, f)

    texts = df["tokens"].apply(str.split).tolist()
    tags_list = df["tags"].tolist()

    # Split before tokenization
    X_train_texts, X_val_texts, y_train_tags, y_val_tags = train_test_split(
        texts,
        tags_list,
        test_size=0.2,
        random_state=42
    )

    # Tokenize
    train_encodings = tokenizer(
        X_train_texts,
        is_split_into_words=True,
        return_offsets_mapping=False,
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

    val_encodings = tokenizer(
        X_val_texts,
        is_split_into_words=True,
        return_offsets_mapping=False,
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

    # Encode labels
    y_train = encode_tags(y_train_tags, train_encodings, label2id)
    y_val = encode_tags(y_val_tags, val_encodings, label2id)

    # Save encodings using pickle
    with open(DATA_DIR / "X_train.pkl", "wb") as f:
        pickle.dump(train_encodings, f)
    with open(DATA_DIR / "X_val.pkl", "wb") as f:
        pickle.dump(val_encodings, f)

    # Save labels
    np.save(DATA_DIR / "y_train.npy", y_train, allow_pickle=True)
    np.save(DATA_DIR / "y_val.npy", y_val, allow_pickle=True)

    print("✅ Preprocessing complete. Encodings and labels saved.")

if __name__ == "__main__":
    preprocess()
