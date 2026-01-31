from predict import NERPredictor

ner = NERPredictor()

text = "Python Java SQL"
tokens, labels = ner.predict(text)

for t, l in zip(tokens, labels):
    print(f"{t:15} {l}")
