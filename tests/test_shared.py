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
        default_params={"hl": "en", "gl": "gb"},
    )

    encoded = runtime.execute(
        engine="google",
        typed_params={"q": "coffee", "num": 3},
    )

    assert client.calls == [{"hl": "en", "gl": "gb", "q": "coffee", "num": 3, "engine": "google"}]
    assert json.loads(encoded)["params"] == client.calls[0]


@pytest.mark.parametrize("reserved", ["api_key", "async", "engine", "output"])
def test_runtime_rejects_transport_parameters_in_defaults(reserved: str) -> None:
    with pytest.raises(ValueError, match=rf"default_params.*{reserved}"):
        SearchRuntime(client=FakeClient(), default_params={reserved: "unsafe"})


def test_runtime_copies_default_params_at_construction() -> None:
    defaults: dict[str, object] = {"hl": "en"}
    runtime = SearchRuntime(client=FakeClient(), default_params=defaults)
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
        ("semantic_kernel", "semantic-kernel"),
        ("google_adk", "google-adk"),
    ],
)
def test_provider_aliases_normalize_without_importing_sdks(alias: str, expected: str) -> None:
    assert normalize_provider(alias) == expected


def test_plain_openai_name_is_not_an_alias_for_openai_agents() -> None:
    assert normalize_provider("openai") == "openai"


def test_auto_provider_detection_uses_documented_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = {"langchain-core", "openai-agents"}

    def fake_version(distribution: str) -> str:
        if distribution in installed:
            return "1.0"
        raise metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(metadata, "version", fake_version)

    assert detect_provider() == "langchain"


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


def test_builtin_client_serializes_access_to_its_requests_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serpapi_module = ModuleType("serpapi")
    state_lock = threading.Lock()
    second_search_entered = threading.Event()
    active_searches = 0
    overlapped = False

    class Client:
        def __init__(self, **kwargs: object) -> None:
            pass

        def search(self, params: dict[str, object]) -> dict[str, object]:
            nonlocal active_searches, overlapped
            with state_lock:
                active_searches += 1
                if active_searches > 1:
                    overlapped = True
                    second_search_entered.set()
            second_search_entered.wait(timeout=0.1)
            with state_lock:
                active_searches -= 1
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

    assert not overlapped


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


def test_mapping_subclass_is_encoded_as_compact_plain_json() -> None:
    class MappingClient:
        def search(self, params: dict[str, object]) -> UserDict[str, object]:
            return UserDict({"params": params, "results": [{"title": "Coffee"}]})

    encoded = SearchRuntime(client=MappingClient()).execute(
        engine="google", typed_params={"q": "coffee"}
    )

    assert json.loads(encoded)["results"] == [{"title": "Coffee"}]
    assert "\n" not in encoded
