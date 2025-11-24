# utils/web_search.py
# Lightweight SerpAPI wrapper. You can replace with your preferred search provider.
import os
import requests
from config.config import SERPAPI_KEY


SERPAPI_URL = "https://serpapi.com/search.json"




def web_search(query: str, num_results: int = 5):
    """Perform live web search using SerpAPI (requires SERPAPI_KEY environment variable).
    Returns a simple list of dicts: title, snippet, link.
    """
    try:
        if not SERPAPI_KEY:
            raise EnvironmentError("SERPAPI_KEY not set in environment")
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "hl": "en",
        }
        r = requests.get(SERPAPI_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = []
        # SerpAPI returns 'organic_results' usually
        for item in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link"),
            })
        return results
    except Exception as e:
        print("Web search failed:", e)
        return []