# Run: uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_search.py  # noqa: E501
"""Call SerpApi tools directly without installing an agent framework."""

from __future__ import annotations

import json

from dotenv import load_dotenv

from serpapi_search_tools import news_search, web_search

load_dotenv()


def main() -> None:
    web = web_search(
        provider="function",
        allowed_engines=["google_light", "bing"],
        default_params={"num": 3, "hl": "en", "gl": "us"},
    )
    news = news_search(
        provider="function",
        default_params={"hl": "en", "gl": "us"},
    )

    web_result = json.loads(web(query="Python packaging"))
    news_result = json.loads(news(query="Python releases"))

    print("Web results:", len(web_result.get("organic_results", [])))
    print("News results:", len(news_result.get("news_results", [])))


if __name__ == "__main__":
    main()
