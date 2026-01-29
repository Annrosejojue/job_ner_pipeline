from predict import NERPredictor

ner = NERPredictor()

text = "Senior Python developer with experience in AWS and NLP"
tokens, labels = ner.predict(text)

for t, l in zip(tokens, labels):
    print(f"{t:15} {l}")
