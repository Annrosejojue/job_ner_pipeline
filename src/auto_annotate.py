import pandas as pd
import spacy
from pathlib import Path
import math

DATA_DIR = Path("data")
CORPUS_PATH = DATA_DIR / "processed" / "corpus.csv"
IOB_PATH = DATA_DIR / "processed" / "corpus_iob.csv"

nlp = spacy.load("en_core_web_sm")

def safe_str(x):
    """Convert NaN/None/float to empty string."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()

def span_to_iob(tokens, span_start, span_end, label):
    tags = ["O"] * len(tokens)
    for i, token in enumerate(tokens):
        if token.idx >= span_start and token.idx < span_end:
            prefix = "B-" if token.idx == span_start else "I-"
            tags[i] = prefix + label
    return tags

def merge_iob_tags(base, new):
    return [b if b != "O" else n for b, n in zip(base, new)]

def annotate_row(text, title, skills_str):
    text = safe_str(text)
    title = safe_str(title)
    skills_str = safe_str(skills_str)

    doc = nlp(text)
    tokens = [t.text for t in doc]
    base_tags = ["O"] * len(tokens)

    # --- JOB TITLE TAGGING ---
    if title:
        idx = text.lower().find(title.lower())
        if idx != -1:
            title_tags = span_to_iob(doc, idx, idx + len(title), "JOB_TITLE")
            base_tags = merge_iob_tags(base_tags, title_tags)

    # --- SKILL TAGGING ---
    skills = [s.strip() for s in skills_str.replace(";", ",").split(",") if s.strip()]

    for skill in skills:
        idx = text.lower().find(skill.lower())
        if idx != -1:
            skill_tags = span_to_iob(doc, idx, idx + len(skill), "SKILL")
            base_tags = merge_iob_tags(base_tags, skill_tags)

    return tokens, base_tags

def build_iob():
    df = pd.read_csv(CORPUS_PATH)
    records = []

    for _, row in df.iterrows():
        text = safe_str(row.get("text"))
        title = safe_str(row.get("title"))
        skills = safe_str(row.get("skills"))

        tokens, tags = annotate_row(text, title, skills)

        records.append({
            "job_id": safe_str(row.get("job_id")),
            "tokens": " ".join(tokens),
            "tags": " ".join(tags),
            "text": text
        })

    out_df = pd.DataFrame(records)
    IOB_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(IOB_PATH, index=False)
    print(f"Saved IOB annotations to {IOB_PATH}")

if __name__ == "__main__":
    build_iob()
