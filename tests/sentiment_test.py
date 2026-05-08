from transformers import pipeline

classifier = pipeline("sentiment-analysis")

texts = [
    "Stock market rises after strong earnings",
    "Markets crash due to inflation fears",
    "Investors remain cautious today"
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}")
    print()
