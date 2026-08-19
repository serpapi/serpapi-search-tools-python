import inspect
import json

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


class ResultClient:
    def search(self, params: dict[str, object]) -> dict[str, object]:
        entries = [{"position": position} for position in range(70)]
        return {
            "answer_box": {"answer": "answer"},
            "knowledge_graph": {"title": "title"},
            "organic_results": entries,
            "related_questions": entries,
            "related_searches": entries,
            "top_stories": entries,
            "news_results": entries,
            "local_results": entries,
            "images_results": entries,
            "shopping_results": entries,
            "video_results": entries,
            "shorts_results": entries,
            "channel_results": entries,
            "playlist_results": entries,
            "movie_results": entries,
            "category_results": entries,
            "properties": entries,
            "best_flights": entries,
            "other_flights": entries,
            "destinations": entries,
        }


USE_FACTORY_DEFAULT = object()


@pytest.mark.parametrize(
    ("factory", "call_kwargs", "default_limit"),
    [
        (web_search, {"query": "coffee"}, 10),
        (news_search, {"query": "coffee"}, 20),
        (maps_search, {"query": "coffee"}, 10),
        (images_search, {"query": "coffee"}, 50),
        (shopping_search, {"query": "coffee"}, 60),
        (videos_search, {"query": "coffee"}, 10),
        (
            hotels_search,
            {
                "query": "hotels in Kyoto",
                "check_in_date": "2030-08-01",
                "check_out_date": "2030-08-04",
            },
            20,
        ),
        (
            flights_search,
            {
                "departure_id": "LAX",
                "arrival_id": "AUS",
                "outbound_date": "2030-08-01",
            },
            10,
        ),
        (travel_explore_search, {"departure_id": "JFK"}, 50),
    ],
)
@pytest.mark.parametrize(
    "result_limit_override",
    [USE_FACTORY_DEFAULT, 3],
    ids=["factory-default", "override"],
)
def test_every_factory_applies_the_result_limit_to_each_retained_list(
    factory,
    call_kwargs: dict[str, object],
    default_limit: int,
    result_limit_override: object,
) -> None:
    factory_kwargs: dict[str, object] = {
        "provider": "function",
        "client": ResultClient(),
        "response_format": SearchResultFormat.JSON,
    }
    if result_limit_override is not USE_FACTORY_DEFAULT:
        factory_kwargs["result_limit"] = result_limit_override
    tool = factory(**factory_kwargs)

    result = json.loads(tool(**call_kwargs))
    retained_lists = [value for value in result.values() if isinstance(value, list)]
    expected_length = (
        default_limit if result_limit_override is USE_FACTORY_DEFAULT else result_limit_override
    )

    assert retained_lists
    assert all(len(value) == expected_length for value in retained_lists)
    assert "result_limit" not in inspect.signature(tool).parameters


@pytest.mark.parametrize("mode", [SearchResultMode.COMPACT, SearchResultMode.FULL])
def test_result_limit_does_not_truncate_nested_hotel_amenities(
    mode: SearchResultMode,
) -> None:
    result_item = {
        "position": 1,
        "title": "Hotel",
        "amenities": ["Pool", "Parking", "Breakfast"],
        "images": [{"thumbnail": "one"}, {"thumbnail": "two"}],
    }

    class HotelsClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return {
                "search_metadata": {"status": "Success"},
                "properties": [{**result_item, "position": position} for position in range(5)],
            }

    tool = hotels_search(
        provider="function",
        client=HotelsClient(),
        mode=mode,
        response_format=SearchResultFormat.JSON,
        result_limit=2,
    )
    result = json.loads(
        tool(
            query="hotels in Kyoto",
            check_in_date="2030-08-01",
            check_out_date="2030-08-04",
        )
    )

    assert len(result["properties"]) == 2
    assert result["properties"][0]["amenities"] == result_item["amenities"]
    if mode is SearchResultMode.FULL:
        assert result["search_metadata"] == {"status": "Success"}


@pytest.mark.parametrize("mode", [SearchResultMode.COMPACT, SearchResultMode.FULL])
def test_result_limit_none_keeps_every_returned_result(mode: SearchResultMode) -> None:
    tool = images_search(
        provider="function",
        client=ResultClient(),
        mode=mode,
        response_format=SearchResultFormat.JSON,
        result_limit=None,
    )

    result = json.loads(tool(query="coffee"))

    assert len(result["images_results"]) == 70


@pytest.mark.parametrize(
    ("engine", "result_key", "link_fields", "expected_link_field", "expected_link"),
    [
        (
            "google_shopping",
            "shopping_results",
            {"product_link": "https://shopping.test/product"},
            "product_link",
            "https://shopping.test/product",
        ),
        (
            "amazon",
            "organic_results",
            {
                "link": "https://amazon.test/tracked",
                "link_clean": "https://amazon.test/product",
                "more_buying_choices_link": "https://amazon.test/offers",
            },
            "link",
            "https://amazon.test/product",
        ),
        (
            "walmart",
            "organic_results",
            {"product_page_url": "https://walmart.test/product"},
            "product_page_url",
            "https://walmart.test/product",
        ),
        (
            "ebay",
            "organic_results",
            {"link": "https://ebay.test/product"},
            "link",
            "https://ebay.test/product",
        ),
    ],
)
def test_compact_shopping_limit_preserves_engine_purchase_links(
    engine: str,
    result_key: str,
    link_fields: dict[str, str],
    expected_link_field: str,
    expected_link: str,
) -> None:
    entries = [
        {
            "position": position,
            "title": f"Result {position}",
            **link_fields,
        }
        for position in range(7)
    ]

    class MultiEngineClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return {
                "search_metadata": {"status": "Success"},
                result_key: entries,
            }

    tool = shopping_search(
        provider="function",
        client=MultiEngineClient(),
        allowed_engines=[engine],
        default_engine=engine,
        response_format=SearchResultFormat.JSON,
        result_limit=3,
    )

    result = json.loads(tool(query="coffee"))

    assert len(result[result_key]) == 3
    assert all(item[expected_link_field] == expected_link for item in result[result_key])
    if engine == "amazon":
        assert all(
            item["more_buying_choices_link"] == "https://amazon.test/offers"
            for item in result[result_key]
        )


@pytest.mark.parametrize(
    ("factory", "factory_kwargs", "expected_params"),
    [
        (
            web_search,
            {
                "allowed_engines": ["duckduckgo"],
                "default_engine": "duckduckgo",
                "default_params": {"m": 20},
            },
            {"m": 20, "q": "coffee", "engine": "duckduckgo"},
        ),
        (
            shopping_search,
            {
                "allowed_engines": ["ebay"],
                "default_engine": "ebay",
                "default_params": {"_ipg": 25},
            },
            {"_ipg": 25, "_nkw": "coffee", "engine": "ebay"},
        ),
    ],
)
def test_result_limit_is_independent_from_valid_upstream_count_parameters(
    factory,
    factory_kwargs: dict[str, object],
    expected_params: dict[str, object],
) -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def search(self, params: dict[str, object]) -> dict[str, object]:
            self.calls.append(params)
            return {"organic_results": [{"position": position} for position in range(10)]}

    client = RecordingClient()
    tool = factory(
        provider="function",
        client=client,
        response_format=SearchResultFormat.JSON,
        result_limit=3,
        **factory_kwargs,
    )

    result = json.loads(tool(query="coffee"))

    assert client.calls == [expected_params]
    assert len(result["organic_results"]) == 3
    assert "result_limit" not in client.calls[0]
