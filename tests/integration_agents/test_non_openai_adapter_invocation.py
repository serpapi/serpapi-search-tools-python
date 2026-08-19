from __future__ import annotations

import asyncio

import pytest
from optional_dependencies import import_optional

from serpapi_search_tools import flights_search, hotels_search

pytestmark = pytest.mark.integration


def test_claude_agent_sdk_handler_invokes_structured_hotel_tool(serpapi_client) -> None:
    import_optional("claude_agent_sdk")
    tool = hotels_search(provider="claude-agent-sdk", client=serpapi_client)

    result = asyncio.run(
        tool.handler(
            {
                "query": "Kyoto hotels",
                "check_in_date": "2026-08-01",
                "check_out_date": "2026-08-04",
            }
        )
    )

    assert serpapi_client.calls[0]["engine"] == "google_hotels"
    assert serpapi_client.calls[0]["check_in_date"] == "2026-08-01"
    output = result["content"][0]["text"]
    assert output.startswith("## Properties")
    assert "Fake result" in output
    assert "search_metadata" not in output


def test_semantic_kernel_function_invokes_structured_flight_tool(serpapi_client) -> None:
    import_optional("semantic_kernel.functions")
    tool = flights_search(provider="semantic-kernel", client=serpapi_client)

    result = tool(departure_id="LAX", arrival_id="AUS", outbound_date="2026-08-01")

    assert serpapi_client.calls[0]["engine"] == "google_flights"
    assert "q" not in serpapi_client.calls[0]
    assert result.startswith("## Best Flights")
    assert "Fake result" in result
    assert "search_metadata" not in result
