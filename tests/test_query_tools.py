import inspect
import json
from enum import Enum

import pytest

from serpapi_search_tools import SearchResultMode
from serpapi_search_tools._query_tools import (
    ShoppingSearchEngine,
    WebSearchEngine,
    images_search,
    maps_search,
    news_search,
    shopping_search,
    videos_search,
    web_search,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(self, params: dict[str, object]) -> dict[str, object]:
        self.calls.append(params)
        return {"params": params}


class FailIfCalledClient:
    def search(self, params: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"client must not be called: {params}")


def test_web_engine_catalog_contains_only_general_web_indexes() -> None:
    assert [engine.value for engine in WebSearchEngine] == [
        "google",
        "google_light",
        "bing",
        "yahoo",
        "duckduckgo",
    ]


def test_shopping_engine_catalog_contains_only_product_searches() -> None:
    assert [engine.value for engine in ShoppingSearchEngine] == [
        "google_shopping",
        "amazon",
        "walmart",
        "ebay",
    ]


@pytest.mark.parametrize(
    ("factory", "engine", "native_query"),
    [
        (web_search, "google", "q"),
        (web_search, "google_light", "q"),
        (web_search, "bing", "q"),
        (web_search, "yahoo", "p"),
        (web_search, "duckduckgo", "q"),
        (shopping_search, "google_shopping", "q"),
        (shopping_search, "amazon", "k"),
        (shopping_search, "walmart", "query"),
        (shopping_search, "ebay", "_nkw"),
    ],
)
def test_multi_engine_tools_route_query_to_native_parameter(
    factory, engine: str, native_query: str
) -> None:
    client = FakeClient()
    tool = factory(
        provider="function",
        allowed_engines=[engine],
        client=client,
        mode=SearchResultMode.FULL,
    )

    payload = json.loads(tool(query="coffee"))

    assert client.calls == [{"engine": engine, native_query: "coffee"}]
    assert payload["params"] == client.calls[0]


def test_web_search_defaults_to_google_light_and_narrows_engine_schema() -> None:
    client = FakeClient()
    tool = web_search(
        provider="function",
        allowed_engines=["google_light", "bing"],
        client=client,
    )
    signature = inspect.signature(tool)
    engine_type = signature.parameters["engine"].annotation

    tool(query="coffee")

    assert issubclass(engine_type, str)
    assert issubclass(engine_type, Enum)
    assert [member.value for member in engine_type] == ["google_light", "bing"]
    assert signature.parameters["engine"].default is engine_type.GOOGLE_LIGHT
    assert client.calls == [{"engine": "google_light", "q": "coffee"}]
    assert "serpapi_params" not in signature.parameters


def test_multi_engine_tool_honors_explicit_engine_and_rejects_disallowed_engine() -> None:
    client = FakeClient()
    tool = web_search(
        provider="function",
        allowed_engines=["google_light", "bing"],
        client=client,
    )

    tool(query="coffee", engine="bing")

    assert client.calls == [{"engine": "bing", "q": "coffee"}]
    with pytest.raises(ValueError, match=r"Engine 'yahoo'.*not allowed"):
        tool(query="coffee", engine="yahoo")
    assert len(client.calls) == 1


def test_multi_engine_tool_honors_custom_default_and_rejects_invalid_default() -> None:
    client = FakeClient()
    tool = web_search(
        provider="function",
        allowed_engines=["google_light", "bing"],
        default_engine="bing",
        client=client,
    )

    tool(query="coffee")

    assert client.calls == [{"engine": "bing", "q": "coffee"}]
    with pytest.raises(ValueError, match=r"Default engine 'yahoo'.*not allowed"):
        web_search(
            provider="function",
            allowed_engines=["google_light", "bing"],
            default_engine="yahoo",
            client=FailIfCalledClient(),
        )


def test_vertical_engine_boundaries_fail_before_calling_client() -> None:
    with pytest.raises(ValueError, match=r"google_news.*news_search"):
        web_search(
            provider="function",
            allowed_engines=["google_news"],
            client=FailIfCalledClient(),
        )
    with pytest.raises(ValueError, match=r"google.*web_search"):
        shopping_search(
            provider="function",
            allowed_engines=["google"],
            client=FailIfCalledClient(),
        )


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (news_search, {"engine": "google_news", "q": "coffee"}),
        (images_search, {"engine": "google_images", "q": "coffee"}),
        (videos_search, {"engine": "youtube", "search_query": "coffee"}),
    ],
)
def test_fixed_query_tools_set_engine_and_native_query(factory, expected) -> None:
    client = FakeClient()
    tool = factory(provider="function", client=client)

    tool(query="coffee")

    assert client.calls == [expected]
    assert "engine" not in inspect.signature(tool).parameters


def test_maps_search_adds_search_mode_and_structured_location() -> None:
    client = FakeClient()
    tool = maps_search(provider="function", client=client)

    tool(query="coffee", location="Austin, Texas", zoom=12, nearby=True)

    assert client.calls == [
        {
            "engine": "google_maps",
            "q": "coffee",
            "type": "search",
            "location": "Austin, Texas",
            "z": 12,
            "nearby": True,
        }
    ]


def test_maps_typed_defaults_override_colliding_constructor_defaults() -> None:
    client = FakeClient()
    tool = maps_search(
        provider="function",
        client=client,
        default_params={"location": "Austin", "z": 12, "nearby": True},
    )

    tool(query="coffee")

    assert client.calls == [{"engine": "google_maps", "q": "coffee", "type": "search"}]


def test_maps_nearby_requires_location_before_client_call() -> None:
    tool = maps_search(provider="function", client=FailIfCalledClient())

    with pytest.raises(ValueError, match="nearby=True requires location"):
        tool(query="coffee", nearby=True)


@pytest.mark.parametrize("zoom", [2, 31, True, 12.5])
def test_maps_zoom_bounds_fail_before_client_call(zoom: object) -> None:
    tool = maps_search(provider="function", client=FailIfCalledClient())

    with pytest.raises(ValueError, match="zoom must be an integer from 3 to 30"):
        tool(query="coffee", location="Austin, Texas", zoom=zoom)


@pytest.mark.parametrize("token_param", ["place_id", "data_cid"])
def test_maps_search_mode_rejects_place_detail_parameters(token_param: str) -> None:
    tool = maps_search(
        provider="function",
        client=FailIfCalledClient(),
        default_params={token_param: "identifier"},
    )

    with pytest.raises(ValueError, match="supports search mode"):
        tool(query="coffee")


@pytest.mark.parametrize(
    "token_param",
    ["topic_token", "kgmid", "publication_token", "section_token", "story_token", "so"],
)
def test_news_query_mode_rejects_advanced_token_parameters(token_param: str) -> None:
    tool = news_search(
        provider="function",
        client=FailIfCalledClient(),
        default_params={token_param: "token"},
    )

    with pytest.raises(ValueError, match="query mode"):
        tool(query="coffee")


@pytest.mark.parametrize(
    ("factory", "default_params", "message"),
    [
        (
            shopping_search,
            {"location": "Austin, Texas", "uule": "encoded"},
            "Google Shopping location and uule",
        ),
        (
            shopping_search,
            {"node": "172282"},
            "Amazon keyword mode cannot be combined with node",
        ),
        (
            images_search,
            {"period_unit": "d", "start_date": "2026-01-01"},
            "relative period filters cannot be combined",
        ),
        (
            images_search,
            {"location": "Austin, Texas", "uule": "encoded"},
            "Google Images location and uule",
        ),
    ],
)
def test_documented_engine_conflicts_fail_before_client_call(
    factory, default_params: dict[str, object], message: str
) -> None:
    kwargs: dict[str, object] = {
        "provider": "function",
        "client": FailIfCalledClient(),
        "default_params": default_params,
    }
    if factory is web_search:
        kwargs["allowed_engines"] = ["bing"]
    if factory is shopping_search:
        kwargs["allowed_engines"] = ["amazon" if "node" in default_params else "google_shopping"]
    tool = factory(**kwargs)

    with pytest.raises(ValueError, match=message):
        tool(query="coffee")


def test_bing_accepts_market_and_country_defaults_together() -> None:
    client = FakeClient()
    tool = web_search(
        provider="function",
        allowed_engines=["bing"],
        client=client,
        default_params={"mkt": "en-US", "cc": "US"},
    )

    tool(query="coffee")

    assert client.calls == [{"mkt": "en-US", "cc": "US", "q": "coffee", "engine": "bing"}]


@pytest.mark.parametrize(
    ("engine", "default_params", "query", "message"),
    [
        ("google", {"location": "Austin", "uule": "encoded"}, "coffee", "Google location"),
        ("google", {"lat": 30.2}, "coffee", "lat and lon must be supplied together"),
        (
            "google_light",
            {"location": "Austin", "uule": "encoded"},
            "coffee",
            "Google Light location and uule",
        ),
        (
            "duckduckgo",
            {"search_assist": True, "m": "us-en"},
            "coffee",
            "search_assist and m",
        ),
        ("duckduckgo", {}, "x" * 501, "500 characters or fewer"),
    ],
)
def test_web_engine_specific_validation_fails_before_client_call(
    engine: str,
    default_params: dict[str, object],
    query: str,
    message: str,
) -> None:
    tool = web_search(
        provider="function",
        allowed_engines=[engine],
        client=FailIfCalledClient(),
        default_params=default_params,
    )

    with pytest.raises(ValueError, match=message):
        tool(query=query)


@pytest.mark.parametrize(
    ("default_params", "call_kwargs", "message"),
    [
        ({"ll": "@30.2,-97.7,12z"}, {"location": "Austin, Texas"}, "location and ll"),
        ({"lat": 30.2}, {}, "lat and lon must be supplied together"),
        (
            {"open_state": "2", "open_on_day": "monday"},
            {},
            "open_state and open_on_day",
        ),
    ],
)
def test_maps_documented_conflicts_fail_before_client_call(
    default_params: dict[str, object],
    call_kwargs: dict[str, object],
    message: str,
) -> None:
    tool = maps_search(
        provider="function",
        client=FailIfCalledClient(),
        default_params=default_params,
    )

    with pytest.raises(ValueError, match=message):
        tool(query="coffee", **call_kwargs)


def test_google_shopping_allows_query_with_shoprs() -> None:
    client = FakeClient()
    tool = shopping_search(
        provider="function",
        allowed_engines=["google_shopping"],
        client=client,
        default_params={"shoprs": "filter-token"},
    )

    tool(query="coffee")

    assert client.calls[0]["q"] == "coffee"
    assert client.calls[0]["shoprs"] == "filter-token"
