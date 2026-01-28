import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw" / "jobs.csv"
OUT_PATH = DATA_DIR / "processed" / "corpus.csv"

def build_corpus():
    df = pd.read_csv(RAW_PATH)
    df.columns = df.columns.str.strip().str.lower()

    texts = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        skills = str(row.get("skills", "")).strip()
        resp = str(row.get("responsibilities", "")).strip()

        text = (
            f"We are hiring a {title}. "
            f"The key skills required are {skills}. "
            f"In this role, you will {resp}."
        )
        texts.append(text)

    out_df = pd.DataFrame({
        "job_id": df.get("jobid", ""),
        "title": df.get("title", ""),
        "skills": df.get("skills", ""),
        "responsibility": df.get("responsibilities", ""),
        "text": texts
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved corpus to {OUT_PATH}")

if __name__ == "__main__":
    build_corpus()
