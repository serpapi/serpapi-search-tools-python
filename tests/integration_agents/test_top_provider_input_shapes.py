from __future__ import annotations

import pytest
from provider_helpers import TOP_AGENT_PROVIDERS, prompt_for_tool, run_agent_provider

pytestmark = pytest.mark.integration

TOOL_CASES = [
    ("web_search", {"query": "coffee", "engine": "yahoo"}, "yahoo"),
    (
        "maps_search",
        {"query": "coffee", "location": "Austin, Texas", "zoom": 12},
        "google_maps",
    ),
    (
        "hotels_search",
        {
            "query": "Kyoto hotels",
            "check_in_date": "2026-08-01",
            "check_out_date": "2026-08-04",
        },
        "google_hotels",
    ),
    (
        "flights_search",
        {"departure_id": "LAX", "arrival_id": "AUS", "outbound_date": "2026-08-01"},
        "google_flights",
    ),
    (
        "travel_explore_search",
        {"departure_id": "JFK", "arrival_area_id": "/m/02j9z"},
        "google_travel_explore",
    ),
]


@pytest.mark.parametrize("provider", TOP_AGENT_PROVIDERS)
@pytest.mark.parametrize(("tool_name", "arguments", "engine"), TOOL_CASES)
def test_top_provider_fake_agent_supports_every_input_shape(
    provider: str,
    tool_name: str,
    arguments: dict[str, object],
    engine: str,
    fake_openai_server,
    serpapi_client,
) -> None:
    run_agent_provider(
        provider,
        base_url=f"{fake_openai_server.url}/v1",
        api_key="fake-key",
        model="gpt-4o-mini",
        prompt=prompt_for_tool(tool_name, arguments),
        tool_names=[tool_name],
        client=serpapi_client,
    )

    assert serpapi_client.calls[0]["engine"] == engine
    if tool_name in {"flights_search", "travel_explore_search"}:
        assert "q" not in serpapi_client.calls[0]
