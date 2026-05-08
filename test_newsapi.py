import requests

api_key = "f8c1ec12f9794e959cb331f42ca3e010"

url = (
    "https://newsapi.org/v2/everything?"
    "q=stock market&"
    "language=en&"
    "sortBy=publishedAt&"
    f"apiKey={api_key}"
)

res = requests.get(url)
data = res.json()

for article in data["articles"][:5]:
    print(article["title"])