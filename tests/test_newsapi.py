import os

import pytest
import requests


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("NEWSAPI_KEY"), reason="NEWSAPI_KEY not set (optional integration test)")
def test_newsapi_lists_titles():
    api_key = os.environ["NEWSAPI_KEY"]
    url = (
        "https://newsapi.org/v2/everything?"
        "q=stock+market&"
        "language=en&"
        "sortBy=publishedAt&"
        f"apiKey={api_key}"
    )
    res = requests.get(url, timeout=30)
    assert res.status_code == 200
    data = res.json()
    articles = data.get("articles") or []
    assert isinstance(articles, list)
