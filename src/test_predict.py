from predict import NERPredictor

ner = NERPredictor()

text = "Develop and execute content strategies; Create SEO-optimized multi-format content; Edit and proofread content for quality and accuracy; Use digital marketing tools to track performance; Leverage AI tools for content creation and optimization; Collaborate with marketing and design teams; Ensure consistency in brand voice; Track content KPIs; Manage content calendars; Mentor junior content writers"
tokens, labels = ner.predict(text)

for t, l in zip(tokens, labels):
    print(f"{t:15} {l}")
