# Run: uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_multi_search.py  # noqa: E501
"""Combine several search verticals and print small, useful summaries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from dotenv import load_dotenv

from serpapi_search_tools import (
    SearchResultFormat,
    maps_search,
    news_search,
    shopping_search,
    web_search,
)

load_dotenv()


def _titles(items: Iterable[Mapping[str, Any]], *, limit: int = 3) -> list[str]:
    return [str(item.get("title") or item.get("name")) for item in list(items)[:limit]]


def main() -> None:
    web = web_search(
        provider="function",
        allowed_engines=["google_light"],
        default_params={"hl": "en", "gl": "us"},
        response_format=SearchResultFormat.JSON,
        result_limit=3,
    )
    news = news_search(
        provider="function",
        default_params={"hl": "en", "gl": "us"},
        response_format=SearchResultFormat.JSON,
    )
    maps = maps_search(
        provider="function",
        default_params={"hl": "en", "gl": "us"},
        response_format=SearchResultFormat.JSON,
    )
    shopping = shopping_search(
        provider="function",
        allowed_engines=["google_shopping"],
        default_params={"hl": "en", "gl": "us"},
        response_format=SearchResultFormat.JSON,
        result_limit=3,
    )

    results = {
        "web": json.loads(web(query="SerpApi Python")),
        "news": json.loads(news(query="SerpApi")),
        "maps": json.loads(maps(query="coffee", location="Austin, Texas", zoom=12)),
        "shopping": json.loads(shopping(query="coffee grinder")),
    }
    summaries = {
        "web": _titles(results["web"].get("organic_results", [])),
        "news": _titles(results["news"].get("news_results", [])),
        "maps": _titles(results["maps"].get("local_results", [])),
        "shopping": _titles(results["shopping"].get("shopping_results", [])),
    }
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
