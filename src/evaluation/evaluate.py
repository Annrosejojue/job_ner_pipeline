import json
from metrics import precision, recall, f1

def evaluate(bert_preds, hybrid_preds, gold):
    p1 = precision(bert_preds, gold)
    r1 = recall(bert_preds, gold)
    f1_bert = f1(p1, r1)

    p2 = precision(hybrid_preds, gold)
    r2 = recall(hybrid_preds, gold)
    f1_hybrid = f1(p2, r2)

    return {
        "bert_precision": p1,
        "bert_recall": r1,
        "bert_f1": f1_bert,
        "hybrid_precision": p2,
        "hybrid_recall": r2,
        "hybrid_f1": f1_hybrid
    }
