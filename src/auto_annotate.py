import pandas as pd
import spacy
from pathlib import Path

DATA_DIR = Path("data")
CORPUS_PATH = DATA_DIR / "processed" / "corpus.csv"
IOB_PATH = DATA_DIR / "processed" / "corpus_iob.csv"

nlp = spacy.load("en_core_web_sm")

def span_to_iob(tokens, span_start, span_end, label):
    tags = ["O"] * len(tokens)
    for i, token in enumerate(tokens):
        if token.idx >= span_start and token.idx < span_end:
            prefix = "B-" if token.idx == span_start else "I-"
            tags[i] = prefix + label
    return tags

def merge_iob_tags(base, new):
    # If base already has a non-O tag, keep it; else use new
    return [b if b != "O" else n for b, n in zip(base, new)]

def annotate_row(text, title, skills_str):
    doc = nlp(text)
    tokens = [t.text for t in doc]
    base_tags = ["O"] * len(tokens)

    # JOB_TITLE tagging
    if isinstance(title, str) and title.strip():
        title = title.strip()
        idx = text.lower().find(title.lower())
        if idx != -1:
            title_tags = span_to_iob(doc, idx, idx + len(title), "JOB_TITLE")
            base_tags = merge_iob_tags(base_tags, title_tags)

    # SKILL tagging (split by ; or ,)
    skills = []
    if isinstance(skills_str, str):
        for part in skills_str.replace(";", ",").split(","):
            s = part.strip()
            if s:
                skills.append(s)

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
        text = row["text"]
        title = row["title"]
        skills = row["skills"]

        tokens, tags = annotate_row(text, title, skills)

        records.append({
            "job_id": row["job_id"],
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
