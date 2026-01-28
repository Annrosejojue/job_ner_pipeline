from lime.lime_text import LimeTextExplainer
import numpy as np
from predict import NERPredictor

def explain_with_lime(text, target_entity="JOB_TITLE"):
    predictor = NERPredictor()

    def predict_proba(texts):
        probs = []
        for t in texts:
            tokens, labels, token_probs = predictor.predict(t)
            entity_ids = [
                i for i, lab in predictor.id2label.items()
                if lab.endswith(target_entity)
            ]
            if not entity_ids:
                probs.append([1.0, 0.0])
                continue
            p = float(token_probs[:, entity_ids].max())
            probs.append([1 - p, p])
        return np.array(probs)

    explainer = LimeTextExplainer(class_names=["NO_ENTITY", target_entity])
    exp = explainer.explain_instance(text, predict_proba, num_features=10)

    print(exp.as_list())

if __name__ == "__main__":
    explain_with_lime("We are hiring a .NET Developer with C# and SQL Server.")
