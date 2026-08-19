import builtins
import json
import sys
import threading
from collections import UserDict
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from types import ModuleType

import pytest

import serpapi_search_tools._shared as shared
from serpapi_search_tools._shared import (
    PROVIDER_ALIASES,
    SearchResultFormat,
    SearchResultMode,
    SearchRuntime,
    detect_provider,
    env_api_key,
    normalize_provider,
    require_nonempty,
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


def test_runtime_merges_parameters_in_documented_precedence_order() -> None:
    client = FakeClient()
    runtime = SearchRuntime(
        client=client,
        default_params={"hl": "en", "gl": "gb", "location": "Austin"},
        mode=SearchResultMode.FULL,
        response_format=SearchResultFormat.JSON,
    )

    encoded = runtime.execute(
        engine="google",
        typed_params={"q": "coffee", "start": 10, "location": None},
    )

    assert client.calls == [
        {"hl": "en", "gl": "gb", "q": "coffee", "start": 10, "engine": "google"}
    ]
    assert json.loads(encoded)["params"] == client.calls[0]


@pytest.mark.parametrize("reserved", ["api_key", "async", "engine", "output"])
def test_runtime_rejects_transport_parameters_in_defaults(reserved: str) -> None:
    with pytest.raises(ValueError, match=rf"default_params.*{reserved}"):
        SearchRuntime(client=FakeClient(), default_params={reserved: "unsafe"})


def test_runtime_copies_default_params_at_construction() -> None:
    defaults: dict[str, object] = {"hl": "en"}
    runtime = SearchRuntime(
        client=FakeClient(),
        default_params=defaults,
        mode=SearchResultMode.FULL,
        response_format=SearchResultFormat.JSON,
    )
    defaults["hl"] = "fr"

    encoded = runtime.execute(
        engine="google_light",
        typed_params={"q": "coffee"},
    )

    assert json.loads(encoded)["params"]["hl"] == "en"


def test_validation_failure_does_not_call_client() -> None:
    runtime = SearchRuntime(client=FailIfCalledClient())

    with pytest.raises(ValueError, match="query must not be empty"):
        runtime.execute(
            engine="google",
            typed_params={"q": ""},
            validator=lambda params: require_nonempty(params, "q", label="query"),
        )


def test_supported_provider_set_matches_public_optional_integrations() -> None:
    assert set(PROVIDER_ALIASES.values()) == {
        "auto",
        "function",
        "langchain",
        "langgraph",
        "crewai",
        "llamaindex",
        "openai-agents",
        "claude-agent-sdk",
        "pydantic-ai",
        "microsoft-agent-framework",
        "autogen",
        "haystack",
        "semantic-kernel",
        "agno",
        "smolagents",
        "google-adk",
    }


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("plain", "function"),
        ("llama_index", "llamaindex"),
        ("openai_agents", "openai-agents"),
        ("anthropic-agent-sdk", "claude-agent-sdk"),
        ("agent_framework", "microsoft-agent-framework"),
        ("maf", "microsoft-agent-framework"),
        ("semantic_kernel", "semantic-kernel"),
        ("google_adk", "google-adk"),
    ],
)
def test_provider_aliases_normalize_without_importing_sdks(alias: str, expected: str) -> None:
    assert normalize_provider(alias) == expected


def test_plain_openai_name_is_not_an_alias_for_openai_agents() -> None:
    assert normalize_provider("openai") == "openai"


def test_auto_provider_detection_rejects_ambiguous_sdk_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = {"langchain-core", "openai-agents"}

    def fake_version(distribution: str) -> str:
        if distribution in installed:
            return "1.0"
        raise metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(metadata, "version", fake_version)

    with pytest.raises(
        RuntimeError,
        match=r"Multiple supported agent SDKs.*langchain.*openai-agents.*provider=",
    ):
        detect_provider()


def test_auto_provider_detection_treats_langgraph_and_langchain_as_one_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = {"langgraph", "langchain-core"}

    def fake_version(distribution: str) -> str:
        if distribution in installed:
            return "1.0"
        raise metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(metadata, "version", fake_version)

    assert detect_provider() == "langgraph"


def test_auto_provider_detection_falls_back_to_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(distribution: str) -> str:
        raise metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(metadata, "version", missing)

    assert detect_provider() == "function"


def test_environment_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPAPI_API_KEY", "preferred")
    monkeypatch.setenv("SERPAPI_KEY", "secondary")
    monkeypatch.setenv("API_KEY", "unrelated-provider-key")
    assert env_api_key() == "preferred"

    monkeypatch.delenv("SERPAPI_API_KEY")
    assert env_api_key() == "secondary"

    monkeypatch.delenv("SERPAPI_KEY")
    assert env_api_key() is None


def test_missing_api_key_fails_before_constructing_builtin_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serpapi_module = ModuleType("serpapi")

    class Client:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError(f"client must not be constructed: {kwargs}")

    serpapi_module.Client = Client
    monkeypatch.setitem(sys.modules, "serpapi", serpapi_module)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Set SERPAPI_API_KEY or SERPAPI_KEY"):
        SearchRuntime().execute(engine="google_light", typed_params={"q": "coffee"})


def test_missing_serpapi_sdk_has_a_direct_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "serpapi":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"pip install serpapi`"):
        SearchRuntime(api_key="passed").execute(
            engine="google_light",
            typed_params={"q": "coffee"},
        )


def test_builtin_client_uses_modern_sdk_and_passes_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    serpapi_module = ModuleType("serpapi")

    class Client:
        def __init__(self, **kwargs: object) -> None:
            calls.append({"init": kwargs})

        def search(self, params: dict[str, object]) -> dict[str, object]:
            calls.append({"search": params})
            return {"params": params}

    serpapi_module.Client = Client
    monkeypatch.setitem(sys.modules, "serpapi", serpapi_module)

    SearchRuntime(
        api_key="passed",
        timeout=2.5,
        response_format=SearchResultFormat.JSON,
    ).execute(
        engine="google_light",
        typed_params={"q": "coffee"},
    )

    assert calls == [
        {"init": {"api_key": "passed", "timeout": 2.5}},
        {"search": {"q": "coffee", "engine": "google_light"}},
    ]


def test_builtin_client_redacts_api_key_from_provider_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_type = getattr(shared, "SerpApiSearchError", None)
    assert error_type is not None
    secret = "super-secret-serpapi-key"
    serpapi_module = ModuleType("serpapi")

    class Client:
        def __init__(self, **kwargs: object) -> None:
            pass

        def search(self, params: dict[str, object]) -> dict[str, object]:
            raise RuntimeError(
                f"400 Client Error for https://serpapi.com/search?q=coffee&api_key={secret}"
            )

    serpapi_module.Client = Client
    monkeypatch.setitem(sys.modules, "serpapi", serpapi_module)

    with pytest.raises(error_type) as caught:
        SearchRuntime(api_key=secret).execute(
            engine="google_light",
            typed_params={"q": "coffee"},
        )

    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert caught.value.__cause__ is None


def test_builtin_client_preserves_sanitized_provider_error_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "super-secret-serpapi-key"
    serpapi_module = ModuleType("serpapi")

    class Response:
        def json(self) -> dict[str, str]:
            return {
                "error": f"Unsupported location; request used api_key={secret}",
            }

    class ProviderError(RuntimeError):
        response = Response()

    class Client:
        def __init__(self, **kwargs: object) -> None:
            pass

        def search(self, params: dict[str, object]) -> dict[str, object]:
            raise ProviderError("400 Client Error")

    serpapi_module.Client = Client
    monkeypatch.setitem(sys.modules, "serpapi", serpapi_module)

    with pytest.raises(shared.SerpApiSearchError) as caught:
        SearchRuntime(api_key=secret).execute(
            engine="google_maps",
            typed_params={"q": "coffee", "location": "unsupported"},
        )

    assert str(caught.value) == (
        "SerpApi request failed: Unsupported location; request used api_key=[REDACTED]"
    )
    assert secret not in repr(caught.value)
    assert caught.value.__cause__ is None


def test_custom_client_failures_do_not_expose_exception_details() -> None:
    secret = "custom-client-secret"

    class FailingClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            raise RuntimeError(f"request failed with token={secret}")

    with pytest.raises(shared.SerpApiSearchError) as caught:
        SearchRuntime(client=FailingClient()).execute(
            engine="google_light",
            typed_params={"q": "coffee"},
        )

    assert str(caught.value) == "Custom search client request failed."
    assert secret not in repr(caught.value)
    assert caught.value.__cause__ is None


def test_runtime_representation_does_not_include_api_key() -> None:
    secret = "super-secret-serpapi-key"

    assert secret not in repr(SearchRuntime(api_key=secret))


def test_builtin_client_is_reused_across_searches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    serpapi_module = ModuleType("serpapi")

    class Client:
        def __init__(self, **kwargs: object) -> None:
            calls.append({"init": kwargs})

        def search(self, params: dict[str, object]) -> dict[str, object]:
            calls.append({"search": params})
            return {"params": params}

    serpapi_module.Client = Client
    monkeypatch.setitem(sys.modules, "serpapi", serpapi_module)
    runtime = SearchRuntime(api_key="passed", response_format=SearchResultFormat.JSON)

    for query in ("coffee", "tea"):
        runtime.execute(
            engine="google_light",
            typed_params={"q": query},
        )

    assert [call for call in calls if "init" in call] == [{"init": {"api_key": "passed"}}]
    assert [call["search"]["q"] for call in calls if "search" in call] == [
        "coffee",
        "tea",
    ]


def test_builtin_client_uses_one_reusable_session_per_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serpapi_module = ModuleType("serpapi")
    state_lock = threading.Lock()
    second_search_entered = threading.Event()
    active_searches = 0
    max_active_searches = 0
    client_ids: set[int] = set()
    active_by_client: dict[int, int] = {}
    max_active_by_client: dict[int, int] = {}

    class Client:
        def __init__(self, **kwargs: object) -> None:
            client_ids.add(id(self))

        def search(self, params: dict[str, object]) -> dict[str, object]:
            nonlocal active_searches, max_active_searches
            client_id = id(self)
            with state_lock:
                active_searches += 1
                max_active_searches = max(max_active_searches, active_searches)
                active_by_client[client_id] = active_by_client.get(client_id, 0) + 1
                max_active_by_client[client_id] = max(
                    max_active_by_client.get(client_id, 0),
                    active_by_client[client_id],
                )
                if active_searches > 1:
                    second_search_entered.set()
            second_search_entered.wait(timeout=0.1)
            with state_lock:
                active_searches -= 1
                active_by_client[client_id] -= 1
            return {"params": params}

    serpapi_module.Client = Client
    monkeypatch.setitem(sys.modules, "serpapi", serpapi_module)
    runtime = SearchRuntime(api_key="passed", response_format=SearchResultFormat.JSON)
    start = threading.Barrier(2)

    def search(query: str) -> None:
        start.wait()
        runtime.execute(
            engine="google_light",
            typed_params={"q": query},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(search, ("coffee", "tea")))

    assert max_active_searches == 2
    assert len(client_ids) == 2
    assert set(max_active_by_client.values()) == {1}


def test_legacy_sdk_collision_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    serpapi_module = ModuleType("serpapi")
    serpapi_module.SerpApiClient = object
    monkeypatch.setitem(sys.modules, "serpapi", serpapi_module)

    with pytest.raises(
        RuntimeError,
        match=r"(?i)google-search-results.*uninstall.*reinstall.*serpapi",
    ):
        SearchRuntime(api_key="passed").execute(engine="google", typed_params={"q": "coffee"})


def test_unsupported_serpapi_module_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "serpapi", ModuleType("serpapi"))

    with pytest.raises(RuntimeError, match=r"does not expose Client\.$"):
        SearchRuntime(api_key="passed").execute(engine="google", typed_params={"q": "coffee"})


def test_mapping_subclass_is_encoded_as_plain_minified_json() -> None:
    class MappingClient:
        def search(self, params: dict[str, object]) -> UserDict[str, object]:
            return UserDict({"params": params, "results": [{"title": "Coffee"}]})

    encoded = SearchRuntime(
        client=MappingClient(),
        mode=SearchResultMode.FULL,
        response_format=SearchResultFormat.JSON,
    ).execute(engine="google", typed_params={"q": "coffee"})

    assert json.loads(encoded)["results"] == [{"title": "Coffee"}]
    assert "\n" not in encoded


def test_search_result_mode_is_a_stable_public_enum() -> None:
    assert [member.value for member in SearchResultMode] == ["compact", "full"]


def test_compact_mode_keeps_web_answers_and_bounds_organic_results() -> None:
    class WebClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return {
                "search_metadata": {"status": "Success"},
                "search_parameters": params,
                "answer_box": {"answer": "Coffee is a brewed drink."},
                "knowledge_graph": {"title": "Coffee"},
                "ai_overview": {"text": "A compact overview."},
                "organic_results": [{"position": position} for position in range(1, 13)],
                "related_searches": [{"query": "tea"}],
                "serpapi_pagination": {"next": "https://example.test/next"},
            }

    encoded = SearchRuntime(client=WebClient(), response_format=SearchResultFormat.JSON).execute(
        engine="google",
        typed_params={"q": "coffee"},
    )
    result = json.loads(encoded)

    assert result == {
        "answer_box": {"answer": "Coffee is a brewed drink."},
        "knowledge_graph": {"title": "Coffee"},
        "ai_overview": {"text": "A compact overview."},
        "organic_results": [{"position": position} for position in range(1, 11)],
    }


@pytest.mark.parametrize(
    ("engine", "expected_keys"),
    [
        (
            "google_light",
            {
                "answer_box",
                "knowledge_graph",
                "organic_results",
                "related_questions",
                "related_searches",
                "top_stories",
            },
        ),
        ("bing", {"answer_box", "knowledge_graph", "copilot_answer", "organic_results"}),
        ("yahoo", {"answer_box", "knowledge_graph", "organic_results"}),
        ("duckduckgo", {"knowledge_graph", "organic_results"}),
        ("google_news", {"news_results"}),
        ("google_maps", {"local_results", "place_results"}),
        ("google_images", {"images_results"}),
        ("google_shopping", {"shopping_results"}),
        ("amazon", {"organic_results"}),
        ("walmart", {"organic_results"}),
        ("ebay", {"organic_results"}),
        (
            "youtube",
            {
                "video_results",
                "shorts_results",
                "channel_results",
                "playlist_results",
                "movie_results",
                "category_results",
            },
        ),
        ("google_hotels", {"properties"}),
        ("google_flights", {"best_flights", "other_flights"}),
        ("google_travel_explore", {"destinations"}),
    ],
)
def test_compact_mode_selects_primary_result_families(
    engine: str,
    expected_keys: set[str],
) -> None:
    all_result_keys = {
        "answer_box",
        "knowledge_graph",
        "ai_overview",
        "copilot_answer",
        "organic_results",
        "news_results",
        "local_results",
        "place_results",
        "images_results",
        "shopping_results",
        "video_results",
        "shorts_results",
        "channel_results",
        "playlist_results",
        "movie_results",
        "category_results",
        "properties",
        "best_flights",
        "other_flights",
        "destinations",
        "related_questions",
        "related_searches",
        "top_stories",
    }

    class ResultClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            result: dict[str, object] = {
                "search_metadata": {"status": "Success"},
                "search_parameters": params,
                "related_searches": [{"query": "noise"}],
            }
            for key in all_result_keys:
                result[key] = (
                    {"value": key}
                    if key
                    in {
                        "answer_box",
                        "knowledge_graph",
                        "ai_overview",
                        "copilot_answer",
                        "place_results",
                    }
                    else [{"value": key}]
                )
            return result

    result = json.loads(
        SearchRuntime(client=ResultClient(), response_format=SearchResultFormat.JSON).execute(
            engine=engine,
            typed_params={"q": "coffee"},
        )
    )

    assert set(result) == expected_keys


def test_google_light_compact_mode_uses_only_supported_result_families() -> None:
    class GoogleLightClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return {
                "ai_overview": {"text": "Not a Google Light response section."},
                "related_questions": [{"question": str(index)} for index in range(12)],
                "related_searches": [{"query": str(index)} for index in range(12)],
                "top_stories": [{"title": str(index)} for index in range(12)],
            }

    result = json.loads(
        SearchRuntime(client=GoogleLightClient(), response_format=SearchResultFormat.JSON).execute(
            engine="google_light",
            typed_params={"q": "coffee"},
        )
    )

    assert "ai_overview" not in result
    assert set(result) == {"related_questions", "related_searches", "top_stories"}
    assert all(len(result[key]) == 10 for key in result)


def test_result_limit_applies_independently_to_each_result_family() -> None:
    class FlightsClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return {
                "best_flights": [{"position": position} for position in range(6)],
                "other_flights": [{"position": position} for position in range(6)],
            }

    result = json.loads(
        SearchRuntime(
            client=FlightsClient(),
            response_format=SearchResultFormat.JSON,
            result_limit=3,
        ).execute(
            engine="google_flights",
            typed_params={"departure_id": "LAX", "arrival_id": "AUS"},
        )
    )

    assert result == {
        "best_flights": [{"position": position} for position in range(3)],
        "other_flights": [{"position": position} for position in range(3)],
    }


@pytest.mark.parametrize(
    ("engine", "result_key", "item", "expected"),
    [
        (
            "google_images",
            "images_results",
            {
                "title": "Latte art",
                "original": "https://images.test/latte.jpg",
                "source_logo": "data:image/png;base64,large",
                "related_content_id": "token",
                "serpapi_related_content_link": "https://serpapi.test/related",
            },
            {"title": "Latte art", "original": "https://images.test/latte.jpg"},
        ),
        (
            "google_shopping",
            "shopping_results",
            {
                "title": "Grinder",
                "price": "$25",
                "product_link": "https://shopping.test/product",
                "immersive_product_page_token": "large-token",
                "serpapi_immersive_product_api": "https://serpapi.test/product",
                "serpapi_thumbnail": "https://serpapi.test/thumbnail",
                "source_icon": "data:image/png;base64,large",
            },
            {
                "title": "Grinder",
                "price": "$25",
                "product_link": "https://shopping.test/product",
            },
        ),
        (
            "amazon",
            "organic_results",
            {
                "title": "Grinder",
                "link": "https://amazon.test/tracked",
                "link_clean": "https://amazon.test/product",
                "serpapi_link": "https://serpapi.test/product",
                "more_buying_choices_link": "https://amazon.test/offers",
                "purchase_options": [{"price": "$25"}],
            },
            {
                "title": "Grinder",
                "link": "https://amazon.test/product",
                "more_buying_choices_link": "https://amazon.test/offers",
            },
        ),
        (
            "walmart",
            "organic_results",
            {
                "title": "Grinder",
                "product_page_url": "https://walmart.test/product",
                "serpapi_product_page_url": "https://serpapi.test/product",
                "seller_id": "seller",
                "variant_swatches": [{"name": "Black"}],
                "muliple_options_available": False,
            },
            {"title": "Grinder", "product_page_url": "https://walmart.test/product"},
        ),
        (
            "ebay",
            "organic_results",
            {
                "title": "Grinder",
                "link": "https://ebay.test/product?tracking=large",
                "serpapi_link": "https://serpapi.test/product",
                "watchers": "18 watchers",
                "extracted_watchers": 18,
                "buying_format": "buy_it_now",
                "buying_format_text": "Buy It Now",
            },
            {
                "title": "Grinder",
                "link": "https://ebay.test/product?tracking=large",
                "extracted_watchers": 18,
                "buying_format": "buy_it_now",
            },
        ),
        (
            "google_hotels",
            "properties",
            {
                "name": "Hotel",
                "rate_per_night": {"lowest": "$100"},
                "images": [{"thumbnail": "one"}, {"thumbnail": "two"}],
                "reviews_breakdown": [{"name": "Rooms"}],
                "nearby_places": [{"name": "Airport"}],
                "serpapi_property_details_link": "https://serpapi.test/property",
                "serpapi_google_hotels_reviews_link": "https://serpapi.test/reviews",
                "serpapi_google_hotels_photos_link": "https://serpapi.test/photos",
            },
            {
                "name": "Hotel",
                "rate_per_night": {"lowest": "$100"},
                "images": [{"thumbnail": "one"}],
            },
        ),
        (
            "google_travel_explore",
            "destinations",
            {
                "name": "Paris",
                "flight_price": 622,
                "link": "https://google.test/travel",
                "serpapi_link": "https://serpapi.test/travel",
            },
            {"name": "Paris", "flight_price": 622, "link": "https://google.test/travel"},
        ),
    ],
)
def test_compact_mode_projects_large_vertical_result_objects(
    engine: str,
    result_key: str,
    item: dict[str, object],
    expected: dict[str, object],
) -> None:
    class ProjectionClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return {result_key: [item]}

    result = json.loads(
        SearchRuntime(client=ProjectionClient(), response_format=SearchResultFormat.JSON).execute(
            engine=engine,
            typed_params={"q": "coffee"},
        )
    )

    assert result == {result_key: [expected]}


def test_compact_mode_returns_bounded_status_when_no_result_family_is_present() -> None:
    class EmptyResultClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return {
                "search_metadata": {
                    "id": "sensitive-debug-id",
                    "status": "Success",
                },
                "search_information": {
                    "query_displayed": "coffee",
                    "total_results": 0,
                    "raw_html_file": "https://example.test/private",
                },
            }

    result = json.loads(
        SearchRuntime(client=EmptyResultClient(), response_format=SearchResultFormat.JSON).execute(
            engine="google_light",
            typed_params={"q": "coffee"},
        )
    )

    assert result == {
        "search_information": {
            "query_displayed": "coffee",
            "total_results": 0,
        },
        "search_metadata": {"status": "Success"},
        "no_results": True,
    }


def test_compact_mode_preserves_returned_api_errors() -> None:
    class ErrorClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return {
                "search_metadata": {"status": "Error"},
                "search_parameters": params,
                "error": "A bounded error message.",
            }

    encoded = SearchRuntime(client=ErrorClient(), response_format=SearchResultFormat.JSON).execute(
        engine="google_light",
        typed_params={"q": "coffee"},
    )

    assert json.loads(encoded) == {"error": "A bounded error message."}


@pytest.mark.parametrize("mode", [SearchResultMode.FULL, "full"])
def test_full_mode_limits_results_but_preserves_every_other_section(
    mode: SearchResultMode | str,
) -> None:
    response = {
        "search_metadata": {"status": "Success"},
        "organic_results": [
            {"title": "Coffee"},
            {"title": "Tea"},
        ],
        "related_searches": [
            {"query": "tea"},
            {"query": "coffee beans"},
        ],
    }

    class FullClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return response

    encoded = SearchRuntime(
        client=FullClient(),
        mode=mode,
        response_format=SearchResultFormat.JSON,
        result_limit=1,
    ).execute(engine="google_light", typed_params={"q": "coffee"})

    assert json.loads(encoded) == {
        "search_metadata": {"status": "Success"},
        "organic_results": [{"title": "Coffee"}],
        "related_searches": [{"query": "tea"}],
    }


def test_invalid_result_mode_fails_when_the_tool_is_created() -> None:
    with pytest.raises(ValueError, match=r"mode must be one of: compact, full"):
        SearchRuntime(client=FakeClient(), mode="summary")


@pytest.mark.parametrize("result_limit", [0, True, "5"])
def test_invalid_result_limit_fails_when_the_tool_is_created(result_limit: object) -> None:
    with pytest.raises(ValueError, match="result_limit must be a positive integer or None"):
        SearchRuntime(client=FakeClient(), result_limit=result_limit)  # type: ignore[arg-type]
