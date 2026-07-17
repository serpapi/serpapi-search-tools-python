from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import pytest

from serpapi_search_tools import (
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

pytestmark = pytest.mark.integration

PROVIDER_IMPORTS = {
    "agno": "agno.tools",
    "autogen": "autogen_core.tools",
    "claude-agent-sdk": "claude_agent_sdk",
    "crewai": "crewai.tools",
    "google-adk": "google.adk.tools",
    "haystack": "haystack.tools",
    "langchain": "langchain_core.tools",
    "langgraph": "langgraph",
    "llamaindex": "llama_index.core.tools",
    "openai-agents": "agents",
    "pydantic-ai": "pydantic_ai",
    "semantic-kernel": "semantic_kernel.functions",
    "smolagents": "smolagents",
}


@dataclass
class _NoSearchClient:
    def search(self, params: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"adapter construction unexpectedly searched with {params!r}")


def test_tox_environment_requires_its_selected_framework() -> None:
    provider = os.getenv("SERPAPI_SEARCH_TOOL_REQUIRED_PROVIDER")
    if provider is None:
        pytest.skip("required-provider guard is enabled by the integration tox environments")

    if provider not in PROVIDER_IMPORTS:
        pytest.fail(f"unknown required provider {provider!r}")

    module_name = PROVIDER_IMPORTS[provider]
    import_module(module_name)

    for factory in _factories():
        assert factory(provider=provider, client=_NoSearchClient()) is not None


def _factories() -> tuple[Any, ...]:
    return (
        web_search,
        news_search,
        maps_search,
        images_search,
        shopping_search,
        videos_search,
        hotels_search,
        flights_search,
        travel_explore_search,
    )
