import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw" / "jobs.csv"      # your Kaggle file
OUT_PATH = DATA_DIR / "processed" / "corpus.csv"

def build_corpus():
    df = pd.read_csv(RAW_PATH)

    # Basic cleaning of NaNs
    df = df.fillna("")

    # Build a synthetic "description" field
    # You can tweak this template later
    texts = []
    for _, row in df.iterrows():
        title = str(row["Title"]).strip()
        skills = str(row["Skills"]).strip()
        resp = str(row["Responsibility"]).strip()

        text = f"We are hiring a {title}. The key skills required are {skills}. " \
               f"In this role, you will {resp}."
        texts.append(text)

    out_df = pd.DataFrame({
        "job_id": df["JobID"],
        "title": df["Title"],
        "skills": df["Skills"],
        "responsibility": df["Responsibility"],
        "text": texts
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved corpus to {OUT_PATH}")

if __name__ == "__main__":
    build_corpus()
