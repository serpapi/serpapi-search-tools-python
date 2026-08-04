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
    )

    encoded = runtime.execute(
        engine="google",
        typed_params={"q": "coffee", "num": 3, "location": None},
    )

    assert client.calls == [{"hl": "en", "gl": "gb", "q": "coffee", "num": 3, "engine": "google"}]
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

    SearchRuntime(api_key="passed", timeout=2.5).execute(
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
    runtime = SearchRuntime(api_key="passed")

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
    runtime = SearchRuntime(api_key="passed")
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

    encoded = SearchRuntime(client=MappingClient(), mode=SearchResultMode.FULL).execute(
        engine="google", typed_params={"q": "coffee"}
    )

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
                "organic_results": [{"position": position} for position in range(1, 9)],
                "related_searches": [{"query": "tea"}],
                "serpapi_pagination": {"next": "https://example.test/next"},
            }

    encoded = SearchRuntime(client=WebClient()).execute(
        engine="google",
        typed_params={"q": "coffee"},
    )
    result = json.loads(encoded)

    assert result == {
        "answer_box": {"answer": "Coffee is a brewed drink."},
        "knowledge_graph": {"title": "Coffee"},
        "ai_overview": {"text": "A compact overview."},
        "organic_results": [{"position": position} for position in range(1, 6)],
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
        ("bing", {"answer_box", "knowledge_graph", "ai_overview", "organic_results"}),
        ("yahoo", {"answer_box", "knowledge_graph", "ai_overview", "organic_results"}),
        ("duckduckgo", {"answer_box", "knowledge_graph", "ai_overview", "organic_results"}),
        ("google_news", {"news_results"}),
        ("google_maps", {"local_results"}),
        ("google_images", {"images_results"}),
        ("google_shopping", {"shopping_results"}),
        ("amazon", {"organic_results"}),
        ("walmart", {"organic_results"}),
        ("ebay", {"organic_results"}),
        ("youtube", {"video_results"}),
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
        "organic_results",
        "news_results",
        "local_results",
        "images_results",
        "shopping_results",
        "video_results",
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
                    }
                    else [{"value": key}]
                )
            return result

    result = json.loads(
        SearchRuntime(client=ResultClient()).execute(
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
                "related_questions": [{"question": str(index)} for index in range(7)],
                "related_searches": [{"query": str(index)} for index in range(7)],
                "top_stories": [{"title": str(index)} for index in range(7)],
            }

    result = json.loads(
        SearchRuntime(client=GoogleLightClient()).execute(
            engine="google_light",
            typed_params={"q": "coffee"},
        )
    )

    assert "ai_overview" not in result
    assert set(result) == {"related_questions", "related_searches", "top_stories"}
    assert all(len(result[key]) == 5 for key in result)


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
        SearchRuntime(client=EmptyResultClient()).execute(
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

    encoded = SearchRuntime(client=ErrorClient()).execute(
        engine="google_light",
        typed_params={"q": "coffee"},
    )

    assert json.loads(encoded) == {"error": "A bounded error message."}


@pytest.mark.parametrize("mode", [SearchResultMode.FULL, "full"])
def test_full_mode_preserves_every_response_section(mode: SearchResultMode | str) -> None:
    response = {
        "search_metadata": {"status": "Success"},
        "organic_results": [{"title": "Coffee"}],
        "related_searches": [{"query": "tea"}],
    }

    class FullClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return response

    encoded = SearchRuntime(
        client=FullClient(),
        mode=mode,
    ).execute(engine="google_light", typed_params={"q": "coffee"})

    assert json.loads(encoded) == response


def test_invalid_result_mode_fails_when_the_tool_is_created() -> None:
    with pytest.raises(ValueError, match=r"mode must be one of: compact, full"):
        SearchRuntime(client=FakeClient(), mode="summary")
