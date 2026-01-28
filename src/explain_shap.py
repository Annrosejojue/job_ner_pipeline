import shap
import numpy as np
from predict import NERPredictor

def explain_with_shap(text, target_entity="JOB_TITLE"):
    predictor = NERPredictor()

    target_label = f"B-{target_entity}"
    target_id = None
    for i, lab in predictor.id2label.items():
        if lab == target_label:
            target_id = i
            break

    def f(texts):
        probs = []
        for t in texts:
            tokens, labels, token_probs = predictor.predict(t)
            p = float(token_probs[:, target_id].max())
            probs.append([1 - p, p])
        return np.array(probs)

    explainer = shap.Explainer(f, shap.maskers.Text(tokenizer=str.split))
    shap_values = explainer([text])
    shap.plots.text(shap_values[0])

if __name__ == "__main__":
    explain_with_shap("We are hiring a .NET Developer with C# and SQL Server.")
