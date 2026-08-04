# Run: uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_cached_search.py  # noqa: E501
"""Wrap the SerpApi client with a tiny in-memory cache and safe logging."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import serpapi
from dotenv import load_dotenv

from serpapi_search_tools import web_search

load_dotenv()


def _serpapi_api_key() -> str:
    value = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not value:
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")
    return value


class CachingClient:
    def __init__(self, api_key: str) -> None:
        self._client = serpapi.Client(api_key=api_key)
        self._cache: dict[str, dict[str, Any]] = {}

    def search(self, params: dict[str, Any]) -> Mapping[str, Any]:
        safe_params = dict(params)
        safe_params.pop("api_key", None)
        cache_key = json.dumps(safe_params, sort_keys=True, default=str)
        if cache_key in self._cache:
            print("cache hit:", safe_params)
            return self._cache[cache_key]

        print("cache miss:", safe_params)
        result = dict(self._client.search(params))
        self._cache[cache_key] = result
        return result


def main() -> None:
    client = CachingClient(_serpapi_api_key())
    search = web_search(
        provider="function",
        client=client,
        allowed_engines=["google_light"],
        default_params={"num": 3, "hl": "en"},
    )

    first = json.loads(search(query="Python packaging"))
    second = json.loads(search(query="Python packaging"))
    print("same result:", first == second)


if __name__ == "__main__":
    main()
