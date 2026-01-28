def find_iob_errors(true_tags, pred_tags):
    errors = []
    for t, p in zip(true_tags, pred_tags):
        if t != p:
            errors.append((t, p))
    return errors
