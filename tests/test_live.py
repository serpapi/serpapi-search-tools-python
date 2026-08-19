from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from serpapi_search_tools import (
    SearchResultFormat,
    SearchResultMode,
    SerpApiSearchError,
    TravelClass,
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
from tests.live_contracts import LIVE_PARAMETER_CASES, LiveParameterCase

pytestmark = pytest.mark.live

if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
    pytest.skip(
        "Set SERPAPI_API_KEY or SERPAPI_KEY before running the live test suite.",
        allow_module_level=True,
    )

UNICODE_QUERY = "café 東京 'best' & #1"
RESPONSE_KEYS: dict[str, list[str]] = {}

ENGINE_CASES = (
    pytest.param("web-google", "google", "q", "SerpApi", "organic_results", id="web-google"),
    pytest.param(
        "web-google-light",
        "google_light",
        "q",
        "SerpApi",
        "organic_results",
        id="web-google-light",
    ),
    pytest.param("web-bing", "bing", "q", "SerpApi", "organic_results", id="web-bing"),
    pytest.param("web-yahoo", "yahoo", "p", "SerpApi", "organic_results", id="web-yahoo-native-p"),
    pytest.param(
        "web-duckduckgo",
        "duckduckgo",
        "q",
        "SerpApi",
        "organic_results",
        id="web-duckduckgo",
    ),
    pytest.param("news", "google_news", "q", "technology", "news_results", id="news"),
    pytest.param("maps", "google_maps", "q", "coffee", "local_results", id="maps"),
    pytest.param("images", "google_images", "q", "latte art", "images_results", id="images"),
    pytest.param(
        "shopping-google",
        "google_shopping",
        "q",
        "coffee grinder",
        "shopping_results",
        id="shopping-google",
    ),
    pytest.param(
        "shopping-amazon",
        "amazon",
        "k",
        "coffee grinder",
        "organic_results",
        id="shopping-amazon-native-k",
    ),
    pytest.param(
        "shopping-walmart",
        "walmart",
        "query",
        "coffee grinder",
        "organic_results",
        id="shopping-walmart-native-query",
    ),
    pytest.param(
        "shopping-ebay",
        "ebay",
        "_nkw",
        "coffee grinder",
        "organic_results",
        id="shopping-ebay-native-nkw",
    ),
    pytest.param("videos", "youtube", "search_query", "latte art", "video_results", id="videos"),
    pytest.param("hotels", "google_hotels", "q", "hotels in Austin", "properties", id="hotels"),
    pytest.param(
        "flights", "google_flights", "departure_id", "LAX", "best_flights", id="flights-no-q"
    ),
    pytest.param(
        "travel-explore",
        "google_travel_explore",
        "departure_id",
        "LAX",
        "destinations",
        id="travel-explore-no-q",
    ),
)

RESULT_FIELD_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "web-google": (("title",), ("link",)),
    "web-google-light": (("title",), ("link",)),
    "web-bing": (("title",), ("link",)),
    "web-yahoo": (("title",), ("link",)),
    "web-duckduckgo": (("title",), ("link",)),
    "news": (("title",), ("link", "stories")),
    "maps": (("title",), ("place_id", "data_id")),
    "images": (("title",), ("original", "thumbnail")),
    "shopping-google": (("title",), ("price", "extracted_price")),
    "shopping-amazon": (("title",), ("link",), ("price", "extracted_price", "prices")),
    "shopping-walmart": (("title",), ("product_page_url", "link"), ("primary_offer", "price")),
    "shopping-ebay": (("title",), ("link",), ("price", "extracted_price")),
    "videos": (("title",), ("link",)),
    "hotels": (("name",), ("property_token", "link", "rate_per_night")),
    "flights": (("price",), ("flights",)),
    "travel-explore": (("name",), ("destination_id", "flight_price")),
}

MARKDOWN_RESULT_HEADINGS: dict[str, tuple[str, ...]] = {
    "web-google": ("Organic Results",),
    "web-google-light": ("Organic Results",),
    "web-bing": ("Organic Results",),
    "web-yahoo": ("Organic Results",),
    "web-duckduckgo": ("Organic Results",),
    "news": ("News Results",),
    "maps": ("Local Results",),
    "images": ("Images Results",),
    "shopping-google": ("Shopping Results",),
    "shopping-amazon": ("Organic Results",),
    "shopping-walmart": ("Organic Results",),
    "shopping-ebay": ("Organic Results",),
    "videos": ("Video Results", "Shorts Results"),
    "hotels": ("Properties",),
    "flights": ("Best Flights", "Other Flights"),
    "travel-explore": ("Destinations",),
}


@pytest.fixture(scope="session", autouse=True)
def _write_response_key_inventory() -> Iterable[None]:
    yield
    destination = os.getenv("SERPAPI_LIVE_KEY_ARTIFACT")
    if destination:
        Path(destination).write_text(json.dumps(RESPONSE_KEYS, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    ("case", "engine", "native_field", "native_value", "result_key"), ENGINE_CASES
)
def test_live_public_tool_request_contracts(
    case: str,
    engine: str,
    native_field: str,
    native_value: str,
    result_key: str,
) -> None:
    encoded = _call_live_case(case)
    result = _decode_success(encoded, case, expect_full=True)
    parameters = result["search_parameters"]

    assert parameters["engine"] == engine
    assert parameters[native_field] == native_value
    if case in {"flights", "travel-explore"}:
        assert "q" not in parameters

    results = _primary_results(result, result_key)
    assert results, f"{case} returned no {result_key}"
    first = results[0]
    assert isinstance(first, dict)
    for alternatives in RESULT_FIELD_GROUPS[case]:
        assert any(first.get(field) not in (None, "", []) for field in alternatives), (
            case,
            alternatives,
            sorted(first),
        )

    if case == "maps":
        assert str(parameters["z"]) == "12"
        assert parameters["nearby"] in {True, "true", "True"}
    elif case == "hotels":
        assert parameters["children"] == 1
        assert str(parameters["children_ages"]) == "7"
    elif case == "flights":
        assert str(parameters["type"]) == "1"
        assert str(parameters["travel_class"]) == "3"
        assert "return_date" in parameters
    elif case == "travel-explore":
        assert parameters["arrival_area_id"] == "/m/0852h"


@pytest.mark.parametrize(
    ("case", "engine", "native_field", "native_value", "result_key"), ENGINE_CASES
)
def test_live_default_compact_markdown_honors_result_limit(
    case: str,
    engine: str,
    native_field: str,
    native_value: str,
    result_key: str,
) -> None:
    del engine, native_field, native_value, result_key
    markdown = _call_live_case(
        case,
        response_format=None,
        mode=SearchResultMode.COMPACT,
        result_limit=2,
    )

    assert isinstance(markdown, str)
    assert markdown.strip()
    assert not markdown.lstrip().startswith("---")
    present_headings = [
        heading for heading in MARKDOWN_RESULT_HEADINGS[case] if f"## {heading}" in markdown
    ]
    assert present_headings, (case, markdown[:500])
    for heading in present_headings:
        row_count = _markdown_table_row_count(markdown, heading)
        assert 1 <= row_count <= 2, (case, heading, row_count)


@pytest.mark.parametrize(
    ("case", "engine", "native_field", "native_value", "result_key"), ENGINE_CASES
)
def test_live_full_markdown_honors_result_limit_and_preserves_context(
    case: str,
    engine: str,
    native_field: str,
    native_value: str,
    result_key: str,
) -> None:
    del engine, native_field, native_value, result_key
    markdown = _call_live_case(
        case,
        response_format=None,
        mode=SearchResultMode.FULL,
        result_limit=2,
    )

    assert isinstance(markdown, str)
    assert markdown.startswith("---\n")
    present_headings = [
        heading for heading in MARKDOWN_RESULT_HEADINGS[case] if f"## {heading}" in markdown
    ]
    assert present_headings, (case, markdown[:500])
    for heading in present_headings:
        row_count = _markdown_table_row_count(markdown, heading)
        assert 1 <= row_count <= 2, (case, heading, row_count)


def test_live_one_way_flight_uses_type_two() -> None:
    outbound, _ = _future_dates()
    encoded = flights_search(
        provider="function",
        mode=SearchResultMode.FULL,
        response_format=SearchResultFormat.JSON,
    )(
        departure_id="lax",
        arrival_id="aus",
        outbound_date=outbound,
        travel_class=TravelClass.ECONOMY,
    )
    result = _decode_success(encoded, "flights-one-way", expect_full=True)
    parameters = result["search_parameters"]

    assert parameters["departure_id"] == "LAX"
    assert parameters["arrival_id"] == "AUS"
    assert str(parameters["type"]) == "2"
    assert "return_date" not in parameters
    assert _primary_results(result, "best_flights")


def test_live_unicode_query_round_trips() -> None:
    encoded = web_search(
        provider="function",
        allowed_engines=["google_light"],
        mode=SearchResultMode.FULL,
        response_format=SearchResultFormat.JSON,
    )(query=UNICODE_QUERY)
    result = _decode_success(encoded, "unicode-google-light", expect_full=True)

    assert result["search_parameters"]["q"] == UNICODE_QUERY
    assert result["organic_results"]


def test_live_exact_maps_place_is_preserved_in_both_compact_formats() -> None:
    json_encoded = maps_search(
        provider="function",
        response_format=SearchResultFormat.JSON,
        result_limit=1,
    )(query="Eiffel Tower")
    json_result = _decode_success(json_encoded, "maps-exact-place")

    assert set(json_result) == {"place_results"}
    assert isinstance(json_result["place_results"], dict)
    assert json_result["place_results"]["title"] == "Eiffel Tower"

    markdown = maps_search(provider="function", result_limit=1)(query="Eiffel Tower")

    assert markdown.startswith("## Place Results")
    assert "Eiffel Tower" in markdown
    assert "search_metadata" not in markdown


@pytest.mark.parametrize("case", LIVE_PARAMETER_CASES, ids=lambda case: case.id)
def test_each_public_tool_returns_nonempty_compact_results_live(
    case: LiveParameterCase,
) -> None:
    outbound, returning = _future_dates()
    arguments = {
        key: outbound if value == "$check_in" else returning if value == "$check_out" else value
        for key, value in case.arguments.items()
    }
    factories: dict[str, Callable[..., Any]] = {
        "web_search": lambda **kwargs: web_search(allowed_engines=["google_light"], **kwargs),
        "news_search": news_search,
        "maps_search": maps_search,
        "images_search": images_search,
        "shopping_search": lambda **kwargs: shopping_search(
            allowed_engines=["google_shopping"], **kwargs
        ),
        "videos_search": videos_search,
        "hotels_search": hotels_search,
        "flights_search": flights_search,
        "travel_explore_search": travel_explore_search,
    }
    tool = factories[case.factory](
        provider="function",
        default_params=case.default_params,
        response_format=SearchResultFormat.JSON,
        result_limit=case.result_limit,
    )
    result = _decode_success(tool(**arguments), f"docs-{case.id}")

    if case.factory == "flights_search":
        primary_results = _primary_results(result, "best_flights")
    else:
        primary_results = _primary_results(result, case.result_key)
    assert primary_results, f"{case.id} returned no {case.result_key}"
    assert len(primary_results) <= case.result_limit


def test_invalid_api_key_returns_an_actionable_error_contract() -> None:
    tool = web_search(
        provider="function",
        allowed_engines=["google_light"],
        api_key="invalid-live-test-key",
    )

    with pytest.raises(SerpApiSearchError) as caught:
        tool(query="SerpApi")

    message = str(caught.value)
    assert message.startswith("SerpApi request failed: ")
    assert "invalid-live-test-key" not in message


@pytest.mark.weekly_live
@pytest.mark.skipif(os.getenv("SERPAPI_WEEKLY") != "1", reason="weekly validator probe")
@pytest.mark.parametrize(
    "params",
    (
        {
            "engine": "google_images",
            "q": "coffee",
            "location": "Austin",
            "uule": "w+CAIQICImU2FuIEZyYW5jaXNjbyxDYWxpZm9ybmlhLFVuaXRlZCBTdGF0ZXM",
        },
        {"engine": "amazon", "k": "coffee", "node": "172282"},
    ),
)
def test_weekly_local_validator_remains_aligned_with_serpapi(params: dict[str, Any]) -> None:
    import serpapi

    api_key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    wire_params = dict(params)
    response = serpapi.Client(api_key=api_key).request(
        "GET", "/search", wire_params, assert_200=False
    )
    result = response.json()

    assert "error" in result


def _decode_success(
    encoded: str,
    case: str,
    *,
    expect_full: bool = False,
) -> dict[str, Any]:
    result = json.loads(encoded)

    assert type(result) is dict
    assert "\\u001b[" not in encoded
    assert "\x1b[" not in encoded
    assert "\n" not in encoded
    assert "error" not in result, result.get("error")
    if expect_full:
        assert isinstance(result["search_metadata"], dict)
        assert result["search_metadata"]["status"] in {"Success", "Cached"}
        assert isinstance(result["search_parameters"], dict)
    else:
        assert "search_metadata" not in result
        assert "search_parameters" not in result
    RESPONSE_KEYS[case] = sorted(result)
    return result


def _primary_results(result: Mapping[str, Any], preferred_key: str) -> list[dict[str, Any]]:
    if preferred_key == "best_flights":
        flights = result.get("best_flights") or result.get("other_flights") or []
        return flights if isinstance(flights, list) else []
    value = result.get(preferred_key, [])
    return value if isinstance(value, list) else []


def _future_dates() -> tuple[str, str]:
    outbound = date.today() + timedelta(days=60)
    returning = outbound + timedelta(days=4)
    return outbound.isoformat(), returning.isoformat()


def _call_live_case(
    case: str,
    *,
    response_format: SearchResultFormat | None = SearchResultFormat.JSON,
    mode: SearchResultMode = SearchResultMode.FULL,
    result_limit: int | None = None,
) -> str:
    outbound, returning = _future_dates()
    options: dict[str, Any] = {
        "provider": "function",
        "mode": mode,
        "result_limit": result_limit,
    }
    if response_format is not None:
        options["response_format"] = response_format
    calls: dict[str, tuple[Any, dict[str, Any]]] = {
        "web-google": (
            web_search(
                allowed_engines=["google"],
                **options,
            ),
            {"query": "SerpApi"},
        ),
        "web-google-light": (
            web_search(
                allowed_engines=["google_light"],
                **options,
            ),
            {"query": "SerpApi"},
        ),
        "web-yahoo": (
            web_search(
                allowed_engines=["yahoo"],
                **options,
            ),
            {"query": "SerpApi"},
        ),
        "web-bing": (
            web_search(
                allowed_engines=["bing"],
                **options,
            ),
            {"query": "SerpApi"},
        ),
        "web-duckduckgo": (
            web_search(
                allowed_engines=["duckduckgo"],
                **options,
            ),
            {"query": "SerpApi"},
        ),
        "news": (
            news_search(**options),
            {"query": "technology"},
        ),
        "maps": (
            maps_search(**options),
            {"query": "coffee", "location": "Austin, Texas", "zoom": 12, "nearby": True},
        ),
        "images": (
            images_search(**options),
            {"query": "latte art"},
        ),
        "shopping-google": (
            shopping_search(
                allowed_engines=["google_shopping"],
                **options,
            ),
            {"query": "coffee grinder"},
        ),
        "shopping-amazon": (
            shopping_search(
                allowed_engines=["amazon"],
                **options,
            ),
            {"query": "coffee grinder"},
        ),
        "shopping-walmart": (
            shopping_search(
                allowed_engines=["walmart"],
                **options,
            ),
            {"query": "coffee grinder"},
        ),
        "shopping-ebay": (
            shopping_search(
                allowed_engines=["ebay"],
                **options,
            ),
            {"query": "coffee grinder"},
        ),
        "videos": (
            videos_search(**options),
            {"query": "latte art"},
        ),
        "hotels": (
            hotels_search(**options),
            {
                "query": "hotels in Austin",
                "check_in_date": outbound,
                "check_out_date": returning,
                "adults": 2,
                "children": 1,
                "children_ages": [7],
            },
        ),
        "flights": (
            flights_search(**options),
            {
                "departure_id": "lax",
                "arrival_id": "aus",
                "outbound_date": outbound,
                "return_date": returning,
                "travel_class": TravelClass.BUSINESS,
            },
        ),
        "travel-explore": (
            travel_explore_search(**options),
            {
                "departure_id": "LAX",
                "arrival_area_id": "/m/0852h",
            },
        ),
    }
    tool, arguments = calls[case]
    return tool(**arguments)


def _markdown_table_row_count(markdown: str, heading: str) -> int:
    section = markdown.split(f"## {heading}", maxsplit=1)[1]
    section = section.split("\n## ", maxsplit=1)[0]
    table_lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    assert len(table_lines) >= 3, (heading, section[:500])
    return len(table_lines) - 2
