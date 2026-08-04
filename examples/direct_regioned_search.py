# Run: uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_regioned_search.py  # noqa: E501
"""Mount two named web tools with independent regional defaults."""

from __future__ import annotations

import json

from dotenv import load_dotenv

from serpapi_search_tools import web_search

load_dotenv()


def _first_title(encoded: str) -> str | None:
    results = json.loads(encoded).get("organic_results", [])
    return results[0].get("title") if results else None


def main() -> None:
    search_us = web_search(
        provider="function",
        allowed_engines=["google_light"],
        default_params={"gl": "us", "hl": "en", "num": 3},
        include_examples=False,
        name="web_search_us",
    )
    search_de = web_search(
        provider="function",
        allowed_engines=["google_light"],
        default_params={"gl": "de", "hl": "de", "num": 3},
        include_examples=False,
        name="web_search_de",
    )

    query = "electric vehicle incentives"
    print("US:", _first_title(search_us(query=query)))
    print("DE:", _first_title(search_de(query=query)))


if __name__ == "__main__":
    main()
