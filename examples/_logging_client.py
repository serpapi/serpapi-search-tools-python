# Shared response-logging helper; run one of the public example files directly.
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import serpapi

PRIMARY_RESULTS_BY_ENGINE = {
    "google": ("organic_results",),
    "google_light": ("organic_results",),
    "bing": ("organic_results",),
    "yahoo": ("organic_results",),
    "duckduckgo": ("organic_results",),
    "google_news": ("news_results",),
    "google_maps": ("local_results",),
    "google_images": ("images_results",),
    "google_shopping": ("shopping_results",),
    "amazon": ("organic_results",),
    "walmart": ("organic_results",),
    "ebay": ("organic_results",),
    "youtube": ("video_results",),
    "google_hotels": ("properties",),
    "google_flights": ("best_flights", "other_flights"),
    "google_travel_explore": ("destinations",),
}


def _compact_result(result: Mapping[str, Any], *, max_results: int) -> dict[str, Any]:
    parameters = result.get("search_parameters", {})
    engine = parameters.get("engine") if isinstance(parameters, Mapping) else None
    compacted = {
        key: result[key]
        for key in ("search_metadata", "search_parameters", "error")
        if key in result
    }
    for key in PRIMARY_RESULTS_BY_ENGINE.get(str(engine), ()):
        value = result.get(key)
        compacted[key] = value[:max_results] if isinstance(value, list) else value
    return compacted


class LoggingClient:
    """SerpApi client wrapper that logs request shape without credentials."""

    def __init__(self, api_key: str, *, max_results: int | None = None) -> None:
        self._client = serpapi.Client(api_key=api_key)
        self._max_results = max_results

    def search(self, params: dict[str, Any]) -> Mapping[str, Any]:
        safe_params = {key: value for key, value in params.items() if key != "api_key"}
        print("SerpApi request:", json.dumps(safe_params, sort_keys=True, default=str))
        result = dict(self._client.search(params))
        if self._max_results is not None:
            result = _compact_result(result, max_results=self._max_results)
        metadata = result.get("search_metadata", {})
        print(
            "SerpApi response:",
            json.dumps(
                {
                    "engine": safe_params.get("engine"),
                    "status": metadata.get("status"),
                    "result_keys": sorted(
                        key
                        for key, value in result.items()
                        if key.endswith("_results") or isinstance(value, list)
                    ),
                },
                sort_keys=True,
            ),
        )
        return result
