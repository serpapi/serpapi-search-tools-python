from __future__ import annotations

from types import SimpleNamespace

import provider_helpers
import pytest
from model_retry import run_until_tool_called


def test_completion_failure_is_propagated_after_the_tool_was_called() -> None:
    client = SimpleNamespace(calls=[])

    def call_then_fail() -> None:
        client.calls.append({"engine": "google_light"})
        raise TimeoutError("the local model timed out writing its final sentence")

    with pytest.raises(TimeoutError, match="final sentence"):
        run_until_tool_called(call_then_fail, client)


def test_failure_before_a_tool_call_is_not_hidden() -> None:
    client = SimpleNamespace(calls=[])

    with pytest.raises(TimeoutError, match="before tool selection"):
        run_until_tool_called(
            lambda: (_ for _ in ()).throw(TimeoutError("before tool selection")),
            client,
        )


def test_smolagents_allows_slow_local_model_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class OpenAIServerModel:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    class ToolCallingAgent:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, prompt: str) -> str:
            return prompt

    fake_smolagents = SimpleNamespace(
        OpenAIServerModel=OpenAIServerModel,
        ToolCallingAgent=ToolCallingAgent,
    )
    monkeypatch.setattr(provider_helpers, "import_optional", lambda name: fake_smolagents)
    monkeypatch.setattr(provider_helpers, "_tools", lambda *args, **kwargs: [])

    provider_helpers._run_smolagents(
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        model="local-model",
        prompt="search",
        tool_names=["web_search"],
        client=object(),
        serpapi_api_key=None,
    )

    assert captured["client_kwargs"] == {"timeout": 120, "max_retries": 0}
