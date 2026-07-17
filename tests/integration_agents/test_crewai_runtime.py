from __future__ import annotations

import os
from types import SimpleNamespace

import provider_helpers
import pytest


def test_crewai_run_disables_tracing_and_telemetry_without_leaking_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class LLM:
        def __init__(self, **kwargs: object) -> None:
            captured["llm"] = kwargs

    class Agent:
        def __init__(self, **kwargs: object) -> None:
            captured["agent"] = kwargs

    class Task:
        def __init__(self, **kwargs: object) -> None:
            captured["task"] = kwargs

    class Crew:
        def __init__(self, **kwargs: object) -> None:
            captured["crew"] = kwargs
            captured["crew_environment"] = {
                "CREWAI_TESTING": os.environ.get("CREWAI_TESTING"),
                "CREWAI_DISABLE_TELEMETRY": os.environ.get("CREWAI_DISABLE_TELEMETRY"),
            }

        def kickoff(self) -> str:
            captured["kickoff_environment"] = {
                "CREWAI_TESTING": os.environ.get("CREWAI_TESTING"),
                "CREWAI_DISABLE_TELEMETRY": os.environ.get("CREWAI_DISABLE_TELEMETRY"),
            }
            return "complete"

    fake_crewai = SimpleNamespace(LLM=LLM, Agent=Agent, Task=Task, Crew=Crew)
    monkeypatch.setattr(provider_helpers, "import_optional", lambda name: fake_crewai)
    monkeypatch.setattr(provider_helpers, "_tools", lambda *args, **kwargs: [])
    monkeypatch.setenv("CREWAI_TESTING", "original")
    monkeypatch.delenv("CREWAI_DISABLE_TELEMETRY", raising=False)

    result = provider_helpers._run_crewai(
        base_url="http://127.0.0.1:1234/v1",
        api_key="local",
        model="local-model",
        prompt="search",
        tool_names=["web_search"],
        client=object(),
        serpapi_api_key=None,
    )

    assert result == "complete"
    assert captured["crew"]["tracing"] is False  # type: ignore[index]
    expected_environment = {
        "CREWAI_TESTING": "true",
        "CREWAI_DISABLE_TELEMETRY": "true",
    }
    assert captured["crew_environment"] == expected_environment
    assert captured["kickoff_environment"] == expected_environment
    assert os.environ["CREWAI_TESTING"] == "original"
    assert "CREWAI_DISABLE_TELEMETRY" not in os.environ
