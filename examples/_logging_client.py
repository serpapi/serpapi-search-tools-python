# Shared response-logging helper; run one of the public example files directly.
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import serpapi


class LoggingClient:
    """SerpApi client wrapper that logs request shape without credentials."""

    def __init__(self, api_key: str) -> None:
        self._client = serpapi.Client(api_key=api_key)

    def search(self, params: dict[str, Any]) -> Mapping[str, Any]:
        safe_params = {key: value for key, value in params.items() if key != "api_key"}
        print("SerpApi request:", json.dumps(safe_params, sort_keys=True, default=str))
        result = dict(self._client.search(params))
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
