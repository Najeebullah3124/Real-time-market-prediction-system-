import requests

# Test Reuters RSS logic
url = "https://www.reutersagency.com/feed/?best-sectors=business&post_type=best"
res = requests.get(url)
print(f"Reuters Status: {res.status_code}")
if res.status_code == 200:
    print("Reuters Feed is accessible")
