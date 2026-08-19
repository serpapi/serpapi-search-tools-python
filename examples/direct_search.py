# Run: uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_search.py  # noqa: E501
"""Compare default Markdown output with explicit JSON output."""

from __future__ import annotations

import json

from dotenv import load_dotenv

from serpapi_search_tools import SearchResultFormat, news_search, web_search

load_dotenv()


def main() -> None:
    web = web_search(
        provider="function",
        allowed_engines=["google_light", "bing"],
        result_limit=3,
    )
    news = news_search(
        provider="function",
        default_params={"hl": "en", "gl": "us"},
        response_format=SearchResultFormat.JSON,
    )

    web_result = web(query="Python packaging")
    news_result = json.loads(news(query="Python releases"))

    print("Default Markdown response:\n")
    print(web_result)
    print("JSON news result count:", len(news_result.get("news_results", [])))


if __name__ == "__main__":
    main()
