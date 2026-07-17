from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

HELPER_DIR = Path(__file__).resolve().parents[1] / "integration_agents"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

PREFERRED_OPENAI_COMPAT_MODEL = "google/gemma-4-e2b"


@dataclass(frozen=True)
class OpenAICompatLlmSettings:
    base_url: str
    api_key: str
    model: str


@dataclass
class RecordingSearchClient:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def search(self, params: dict[str, Any]) -> Mapping[str, Any]:
        call = dict(params)
        self.calls.append(call)
        return {
            "search_metadata": {"status": "Success"},
            "params": call,
            "organic_results": [{"title": "Fake OpenAI-compatible LLM result"}],
        }


@dataclass
class RecordingLiveSearchClient:
    api_key: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def search(self, params: dict[str, Any]) -> Mapping[str, Any]:
        call = dict(params)
        response = httpx.get(
            "https://serpapi.com/search.json",
            params={**call, "api_key": self.api_key},
            timeout=30,
        )
        response.raise_for_status()
        self.calls.append(call)
        return response.json()


@pytest.fixture(scope="session")
def openai_compat_llm_settings() -> OpenAICompatLlmSettings:
    if not _truthy(os.getenv("RUN_OPENAI_COMPAT_LLM_TESTS")):
        pytest.skip("Set RUN_OPENAI_COMPAT_LLM_TESTS=1 to run OpenAI-compatible LLM tests.")

    base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
    api_key = os.getenv("OPENAI_COMPAT_API_KEY", "local-openai-compatible-server")
    configured_model = os.getenv("OPENAI_COMPAT_MODEL")

    if configured_model:
        return OpenAICompatLlmSettings(
            base_url=base_url,
            api_key=api_key,
            model=configured_model,
        )

    try:
        response = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.fail(
            "The OpenAI-compatible API is unavailable. Set OPENAI_COMPAT_MODEL to skip "
            f"model discovery. Details: {exc}"
        )

    model = _preferred_gemma_model(_model_ids(response.json()))
    if model is None:
        pytest.fail(
            f"No usable model was discovered. Set OPENAI_COMPAT_MODEL; local servers may use "
            f"{PREFERRED_OPENAI_COMPAT_MODEL}."
        )

    return OpenAICompatLlmSettings(base_url=base_url, api_key=api_key, model=model)


@pytest.fixture
def fake_serpapi_client() -> RecordingSearchClient:
    return RecordingSearchClient()


@pytest.fixture
def live_serpapi_client() -> RecordingLiveSearchClient:
    api_key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not api_key:
        pytest.skip("Set SERPAPI_API_KEY or SERPAPI_KEY to run live SerpApi tests.")
    return RecordingLiveSearchClient(api_key=api_key)


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _model_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    data = payload.get("data", [])
    model_ids: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                model_ids.append(item["id"])
    return tuple(model_ids)


def _preferred_gemma_model(models: tuple[str, ...]) -> str | None:
    for model in models:
        if model.lower() == PREFERRED_OPENAI_COMPAT_MODEL:
            return model
    for model in models:
        normalized = model.lower()
        if "gemma" in normalized and "e2b" in normalized:
            return model
    return None
