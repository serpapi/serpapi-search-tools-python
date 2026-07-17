# Run: uv run --with . examples/direct_marketplace_comparison.py
"""Normalize first-page product results from four marketplace engines."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv

from serpapi_search_tools import shopping_search

load_dotenv()

ENGINES = ("google_shopping", "amazon", "walmart", "ebay")
RESULT_FAMILY = {
    "google_shopping": "shopping_results",
    "amazon": "organic_results",
    "walmart": "organic_results",
    "ebay": "organic_results",
}


def _first_product(engine: str, result: Mapping[str, Any]) -> dict[str, Any]:
    items = result.get(RESULT_FAMILY[engine], [])
    item = items[0] if isinstance(items, list) and items else {}
    offer = item.get("primary_offer", {}) if isinstance(item, dict) else {}
    price = item.get("price") or item.get("extracted_price") or offer.get("offer_price")
    if isinstance(price, Mapping):
        price = price.get("raw") or price.get("extracted")
    return {
        "engine": engine,
        "title": item.get("title"),
        "price": price,
        "link": item.get("link") or item.get("product_link") or item.get("product_page_url"),
    }


def main() -> None:
    products = []
    for engine in ENGINES:
        search = shopping_search(
            provider="function",
            allowed_engines=[engine],
            default_params={"num": 3},
        )
        result = json.loads(search(query="noise cancelling headphones"))
        products.append(_first_product(engine, result))

    print(json.dumps(products, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
