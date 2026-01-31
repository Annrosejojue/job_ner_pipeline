def precision(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)
    return len(pred_set & gold_set) / max(len(pred_set), 1)

def recall(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)
    return len(pred_set & gold_set) / max(len(gold_set), 1)

def f1(p, r):
    return 2 * p * r / max((p + r), 1)
