from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from serpapi_search_tools import (
    SearchResultFormat,
    SearchResultMode,
    flights_search,
    hotels_search,
    images_search,
    maps_search,
    news_search,
    shopping_search,
    travel_explore_search,
    videos_search,
    web_search,
)
from serpapi_search_tools._shared import SearchRuntime


class MarkdownClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def search(self, params: dict[str, Any]) -> str:
        self.calls.append(params)
        return self.response


class JsonClient:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def search(self, params: dict[str, Any]) -> Mapping[str, Any]:
        self.calls.append(params)
        return self.response


@pytest.mark.parametrize(
    ("factory", "factory_kwargs", "call_kwargs", "heading"),
    [
        (web_search, {"allowed_engines": ["google_light"]}, {"query": "coffee"}, "Organic Results"),
        (news_search, {}, {"query": "coffee"}, "News Results"),
        (
            maps_search,
            {},
            {"query": "coffee", "location": "Austin, Texas"},
            "Local Results",
        ),
        (images_search, {}, {"query": "coffee"}, "Images Results"),
        (
            shopping_search,
            {"allowed_engines": ["google_shopping"]},
            {"query": "coffee"},
            "Shopping Results",
        ),
        (videos_search, {}, {"query": "coffee"}, "Video Results"),
        (
            hotels_search,
            {},
            {
                "query": "Austin hotels",
                "check_in_date": "2030-08-01",
                "check_out_date": "2030-08-04",
            },
            "Properties",
        ),
        (
            flights_search,
            {},
            {
                "departure_id": "LAX",
                "arrival_id": "AUS",
                "outbound_date": "2030-08-01",
            },
            "Best Flights",
        ),
        (
            travel_explore_search,
            {},
            {"departure_id": "JFK"},
            "Destinations",
        ),
    ],
)
def test_every_factory_requests_and_returns_markdown_by_default(
    factory: Callable[..., Any],
    factory_kwargs: dict[str, Any],
    call_kwargs: dict[str, Any],
    heading: str,
) -> None:
    response = f"## {heading}\n\n| Title |\n| --- |\n| Coffee |\n"
    client = MarkdownClient(response)
    tool = factory(provider="function", client=client, **factory_kwargs)

    result = tool(**call_kwargs)

    assert result == response
    assert client.calls[0]["output"] == "md"


def test_json_is_an_explicit_opt_in_and_keeps_the_existing_compact_contract() -> None:
    client = JsonClient(
        {
            "search_metadata": {"status": "Success"},
            "organic_results": [{"title": "Coffee"}],
        }
    )
    tool = web_search(
        provider="function",
        allowed_engines=["google_light"],
        client=client,
        response_format=SearchResultFormat.JSON,
    )

    result = json.loads(tool(query="coffee"))

    assert result == {"organic_results": [{"title": "Coffee"}]}
    assert "output" not in client.calls[0]


def test_compact_markdown_keeps_result_sections_and_limits_each_table() -> None:
    response = """---
engine: google_flights
---
## Search Information

Private request metadata.

## Best Flights

| Route | Price |
| --- | --- |
| LAX to AUS | 100 |
| LAX to AUS | 120 |
| LAX to AUS | 140 |

## Other Flights

| Route | Price |
| --- | --- |
| LAX to AUS | 90 |
| LAX to AUS | 110 |
| LAX to AUS | 130 |

## Price Insights

Prices are typical.
"""
    runtime = SearchRuntime(client=MarkdownClient(response), result_limit=2)

    result = runtime.execute(engine="google_flights", typed_params={})

    assert "engine: google_flights" not in result
    assert "Search Information" not in result
    assert "Price Insights" not in result
    assert "| LAX to AUS | 100 |" in result
    assert "| LAX to AUS | 120 |" in result
    assert "| LAX to AUS | 140 |" not in result
    assert "| LAX to AUS | 90 |" in result
    assert "| LAX to AUS | 110 |" in result
    assert "| LAX to AUS | 130 |" not in result


def test_compact_markdown_with_no_limit_keeps_every_result_row() -> None:
    response = """## News Results

| Title |
| --- |
| First story |
| Second story |
"""
    runtime = SearchRuntime(client=MarkdownClient(response), result_limit=None)

    result = runtime.execute(engine="google_news", typed_params={"q": "technology"})

    assert "| First story |" in result
    assert "| Second story |" in result


def test_full_markdown_limits_results_but_preserves_supporting_sections() -> None:
    response = """---
engine: google
---
## Organic Results

| Title |
| --- |
| Coffee |
| Tea |

## Related Searches

Coffee beans
"""
    runtime = SearchRuntime(
        client=MarkdownClient(response),
        mode=SearchResultMode.FULL,
        result_limit=1,
    )

    result = runtime.execute(engine="google", typed_params={"q": "coffee"})

    assert result.startswith("---\nengine: google\n---\n")
    assert "| Coffee |" in result
    assert "| Tea |" not in result
    assert "## Related Searches\n\nCoffee beans" in result


@pytest.mark.parametrize("mode", [SearchResultMode.COMPACT, SearchResultMode.FULL])
def test_markdown_result_limit_does_not_truncate_nested_detail_tables(
    mode: SearchResultMode,
) -> None:
    response = """---
engine: google_flights
---
## Best Flights

| Route | Price |
| --- | --- |
| First itinerary | 100 |
| Second itinerary | 120 |
| Third itinerary | 140 |

### First itinerary legs

| Leg | From | To |
| --- | --- | --- |
| 1 | LAX | DFW |
| 2 | DFW | LHR |
| 3 | LHR | CDG |
"""
    runtime = SearchRuntime(
        client=MarkdownClient(response),
        mode=mode,
        result_limit=2,
    )

    result = runtime.execute(engine="google_flights", typed_params={})

    assert "| First itinerary | 100 |" in result
    assert "| Second itinerary | 120 |" in result
    assert "| Third itinerary | 140 |" not in result
    assert "| 1 | LAX | DFW |" in result
    assert "| 2 | DFW | LHR |" in result
    assert "| 3 | LHR | CDG |" in result


def test_full_markdown_with_no_limit_is_returned_byte_for_byte() -> None:
    response = "---\nengine: google\n---\n## Organic Results\n\nOriginal spacing.\n"
    runtime = SearchRuntime(
        client=MarkdownClient(response),
        mode=SearchResultMode.FULL,
        result_limit=None,
    )

    assert runtime.execute(engine="google", typed_params={"q": "coffee"}) == response


def test_compact_markdown_preserves_exact_place_results_without_row_truncation() -> None:
    response = """## Place Results

| Field | Value |
| --- | --- |
| Title | Eiffel Tower |
| Address | Paris |

## Search Information

Metadata
"""
    runtime = SearchRuntime(client=MarkdownClient(response), result_limit=1)

    result = runtime.execute(engine="google_maps", typed_params={"q": "Eiffel Tower"})

    assert (
        result
        == """## Place Results

| Field | Value |
| --- | --- |
| Title | Eiffel Tower |
| Address | Paris |
"""
    )


def test_compact_json_preserves_exact_place_results() -> None:
    place = {
        "title": "Eiffel Tower",
        "address": "5 Avenue Anatole France, Paris",
        "place_id": "ChIJLU7jZClu5kcR4PcOOO6p3I0",
    }
    runtime = SearchRuntime(
        client=JsonClient(
            {
                "search_metadata": {"status": "Success"},
                "place_results": place,
            }
        ),
        response_format=SearchResultFormat.JSON,
        result_limit=1,
    )

    result = json.loads(runtime.execute(engine="google_maps", typed_params={"q": "Eiffel Tower"}))

    assert result == {"place_results": place}


def test_compact_markdown_drops_followup_only_image_id_but_keeps_image_urls() -> None:
    response = """## Images Results

| Title | Thumbnail | Related Content Id | Original |
| --- | --- | --- | --- |
| Latte | ![Thumbnail](https://images.test/thumb.jpg) | follow-up-token | \
https://images.test/original.jpg |
"""
    runtime = SearchRuntime(client=MarkdownClient(response))

    result = runtime.execute(engine="google_images", typed_params={"q": "latte"})

    assert "Related Content Id" not in result
    assert "follow-up-token" not in result
    assert "https://images.test/thumb.jpg" in result
    assert "https://images.test/original.jpg" in result


def test_compact_markdown_preserves_api_errors_instead_of_returning_blank_text() -> None:
    response = """---
engine: google_light
---
## Search Information

No results.

## Error

Google Light returned no results for this query.
"""
    runtime = SearchRuntime(client=MarkdownClient(response))

    result = runtime.execute(engine="google_light", typed_params={"q": "missing"})

    assert (
        result
        == """## Error

Google Light returned no results for this query.
"""
    )


def test_compact_markdown_falls_back_to_unknown_server_sections() -> None:
    response = """---
engine: google
---
## New Result Family

Server data that the installed library does not know yet.
"""
    runtime = SearchRuntime(client=MarkdownClient(response))

    result = runtime.execute(engine="google", typed_params={"q": "coffee"})

    assert (
        result
        == """## New Result Family

Server data that the installed library does not know yet.
"""
    )


@pytest.mark.parametrize(
    ("response_format", "response", "message"),
    [
        (SearchResultFormat.MARKDOWN, {"organic_results": []}, "must return str"),
        (SearchResultFormat.JSON, "## Organic Results", "must return a mapping"),
    ],
)
def test_runtime_rejects_client_responses_that_do_not_match_the_selected_format(
    response_format: SearchResultFormat,
    response: Mapping[str, Any] | str,
    message: str,
) -> None:
    client = JsonClient(response) if isinstance(response, Mapping) else MarkdownClient(response)
    runtime = SearchRuntime(client=client, response_format=response_format)

    with pytest.raises(TypeError, match=message):
        runtime.execute(engine="google", typed_params={"q": "coffee"})


def test_invalid_response_format_fails_when_the_tool_is_created() -> None:
    with pytest.raises(ValueError, match="response_format must be one of: markdown, json"):
        SearchRuntime(client=MarkdownClient(""), response_format="yaml")
