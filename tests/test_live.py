from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from serpapi_search_tools import (
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
    "news": (("title",), ("link",)),
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
    result = _decode_success(encoded, case)
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


def test_live_one_way_flight_uses_type_two() -> None:
    outbound, _ = _future_dates()
    encoded = flights_search(provider="function")(
        departure_id="lax",
        arrival_id="aus",
        outbound_date=outbound,
        travel_class=TravelClass.ECONOMY,
    )
    result = _decode_success(encoded, "flights-one-way")
    parameters = result["search_parameters"]

    assert parameters["departure_id"] == "LAX"
    assert parameters["arrival_id"] == "AUS"
    assert str(parameters["type"]) == "2"
    assert "return_date" not in parameters
    assert _primary_results(result, "best_flights")


def test_live_unicode_query_round_trips() -> None:
    encoded = web_search(provider="function", allowed_engines=["google_light"])(query=UNICODE_QUERY)
    result = _decode_success(encoded, "unicode-google-light")

    assert result["search_parameters"]["q"] == UNICODE_QUERY
    assert result["organic_results"]


@pytest.mark.parametrize("case", LIVE_PARAMETER_CASES, ids=lambda case: case.id)
def test_representative_default_parameter_combinations_work_live(
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
        "hotels_search": hotels_search,
        "flights_search": flights_search,
        "travel_explore_search": travel_explore_search,
    }
    tool = factories[case.factory](provider="function", default_params=case.default_params)
    result = _decode_success(tool(**arguments), f"docs-{case.id}")

    if case.factory == "flights_search":
        assert _primary_results(result, "best_flights")
    else:
        assert result.get(case.result_key), f"{case.id} returned no {case.result_key}"
    if case.factory == "travel_explore_search":
        parameters = result["search_parameters"]
        assert str(parameters["type"]) == "1"
        assert parameters["outbound_date"] == outbound
        assert parameters["return_date"] == returning


def test_invalid_api_key_returns_an_actionable_error_contract() -> None:
    tool = web_search(
        provider="function",
        allowed_engines=["google_light"],
        api_key="invalid-live-test-key",
    )

    with pytest.raises(SerpApiSearchError) as caught:
        tool(query="SerpApi")

    assert "401" in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "invalid-live-test-key" not in str(caught.value)


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


def _decode_success(encoded: str, case: str) -> dict[str, Any]:
    result = json.loads(encoded)

    assert type(result) is dict
    assert "\\u001b[" not in encoded
    assert "\x1b[" not in encoded
    assert "\n" not in encoded
    assert "error" not in result, result.get("error")
    assert isinstance(result["search_metadata"], dict)
    assert result["search_metadata"]["status"] in {"Success", "Cached"}
    assert isinstance(result["search_parameters"], dict)
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


def _call_live_case(case: str) -> str:
    outbound, returning = _future_dates()
    calls: dict[str, tuple[Any, dict[str, Any]]] = {
        "web-google": (
            web_search(provider="function", allowed_engines=["google"]),
            {"query": "SerpApi"},
        ),
        "web-google-light": (
            web_search(provider="function", allowed_engines=["google_light"]),
            {"query": "SerpApi"},
        ),
        "web-yahoo": (
            web_search(provider="function", allowed_engines=["yahoo"]),
            {"query": "SerpApi"},
        ),
        "web-bing": (
            web_search(provider="function", allowed_engines=["bing"]),
            {"query": "SerpApi"},
        ),
        "web-duckduckgo": (
            web_search(provider="function", allowed_engines=["duckduckgo"]),
            {"query": "SerpApi"},
        ),
        "news": (news_search(provider="function"), {"query": "technology"}),
        "maps": (
            maps_search(provider="function"),
            {"query": "coffee", "location": "Austin, Texas", "zoom": 12, "nearby": True},
        ),
        "images": (images_search(provider="function"), {"query": "latte art"}),
        "shopping-google": (
            shopping_search(provider="function", allowed_engines=["google_shopping"]),
            {"query": "coffee grinder"},
        ),
        "shopping-amazon": (
            shopping_search(provider="function", allowed_engines=["amazon"]),
            {"query": "coffee grinder"},
        ),
        "shopping-walmart": (
            shopping_search(provider="function", allowed_engines=["walmart"]),
            {"query": "coffee grinder"},
        ),
        "shopping-ebay": (
            shopping_search(provider="function", allowed_engines=["ebay"]),
            {"query": "coffee grinder"},
        ),
        "videos": (videos_search(provider="function"), {"query": "latte art"}),
        "hotels": (
            hotels_search(provider="function"),
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
            flights_search(provider="function"),
            {
                "departure_id": "lax",
                "arrival_id": "aus",
                "outbound_date": outbound,
                "return_date": returning,
                "travel_class": TravelClass.BUSINESS,
            },
        ),
        "travel-explore": (
            travel_explore_search(provider="function"),
            {
                "departure_id": "LAX",
                "arrival_area_id": "/m/0852h",
            },
        ),
    }
    tool, arguments = calls[case]
    return tool(**arguments)
