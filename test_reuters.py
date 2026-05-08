import feedparser

url = "https://feeds.feedburner.com/reuters/businessNews"

feed = feedparser.parse(url)

print("Feed Title:", feed.feed.title)
print()

for entry in feed.entries[:5]:
    print(entry.title)