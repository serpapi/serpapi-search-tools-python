"""Smoke-test the built wheel with only its declared runtime dependencies."""

from __future__ import annotations

import json
from importlib import metadata
from typing import Any

import serpapi
import serpapi.models

from serpapi_search_tools import flights_search, web_search

AGENT_SDK_DISTRIBUTIONS = (
    "agno",
    "autogen-core",
    "claude-agent-sdk",
    "crewai",
    "google-adk",
    "haystack-ai",
    "langchain-core",
    "langgraph",
    "llama-index-core",
    "agent-framework-core",
    "openai-agents",
    "pydantic-ai",
    "semantic-kernel",
    "smolagents",
)


class _Response:
    text = ""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def json(self) -> dict[str, Any]:
        return self._result


def _fake_request(
    client: object,
    method: str,
    path: str,
    *,
    params: dict[str, Any],
    **kwargs: Any,
) -> _Response:
    assert client.__class__ is serpapi.Client
    assert method == "GET"
    assert path == "/search"
    assert params in (
        {"engine": "google_light", "q": "coffee"},
        {
            "engine": "google_flights",
            "departure_id": "LAX",
            "arrival_id": "AUS",
            "outbound_date": "2026-08-01",
            "type": 2,
            "travel_class": 1,
            "adults": 1,
            "children": 0,
            "infants_in_seat": 0,
            "infants_on_lap": 0,
        },
    )
    assert kwargs == {}
    if params["engine"] == "google_flights":
        return _Response(
            {
                "search_metadata": {"status": "Success"},
                "best_flights": [{"price": 200}],
                "price_insights": {"lowest_price": 180},
            }
        )
    return _Response(
        {
            "search_metadata": {"status": "Success"},
            "organic_results": [{"title": "Coffee"}],
            "related_searches": [{"query": "tea"}],
        }
    )


def main() -> None:
    for distribution_name in AGENT_SDK_DISTRIBUTIONS:
        try:
            metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue
        raise AssertionError(
            f"clean-install smoke unexpectedly contains agent SDK {distribution_name!r}"
        )

    assert hasattr(serpapi, "Client"), "declared serpapi dependency must expose Client"

    serpapi.Client.request = _fake_request
    serpapi.models.prettify_json = lambda value: f"\x1b[31m{value}\x1b[0m"

    tool = web_search(
        allowed_engines=["google_light"],
        api_key="not-a-real-key",
    )
    assert callable(tool), "provider='auto' must fall back to a plain callable"

    encoded = tool(query="coffee")
    decoded = json.loads(encoded)

    assert type(decoded) is dict
    assert decoded == {
        "organic_results": [{"title": "Coffee"}],
        "related_searches": [{"query": "tea"}],
    }
    assert not encoded.startswith('"')
    assert "\\u001b[" not in encoded
    assert "\x1b[" not in encoded
    assert "\n" not in encoded

    flight = flights_search(provider="function", api_key="not-a-real-key")
    flight_encoded = flight(
        departure_id="LAX",
        arrival_id="AUS",
        outbound_date="2026-08-01",
    )
    assert json.loads(flight_encoded) == {"best_flights": [{"price": 200}]}


if __name__ == "__main__":
    main()
