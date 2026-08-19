from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest

from serpapi_search_tools import SearchResultFormat, web_search

pytestmark = pytest.mark.live

if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
    pytest.skip(
        "Set SERPAPI_API_KEY or SERPAPI_KEY before running live adapter tests.",
        allow_module_level=True,
    )

PROVIDER = os.getenv("SERPAPI_SEARCH_TOOL_REQUIRED_PROVIDER")
if not PROVIDER:
    pytest.skip(
        "Run live adapter tests through a provider-specific integration tox environment.",
        allow_module_level=True,
    )


def test_native_sdk_adapter_executes_a_real_google_light_search() -> None:
    tool = web_search(
        provider=PROVIDER,
        allowed_engines=["google_light"],
        default_params={"hl": "en", "gl": "us"},
        response_format=SearchResultFormat.JSON,
        result_limit=3,
    )

    encoded = _invoke_native(PROVIDER, tool, {"query": "SerpApi Python"})
    result = json.loads(encoded)

    assert "\\u001b[" not in encoded
    assert "\x1b[" not in encoded
    assert "\n" not in encoded
    assert "search_metadata" not in result
    assert "search_parameters" not in result
    assert result["organic_results"]
    assert len(result["organic_results"]) <= 3
    first = result["organic_results"][0]
    assert first["title"]
    assert first["link"]


def _invoke_native(provider: str, tool: Any, arguments: dict[str, Any]) -> str:
    if provider in {"langchain", "langgraph"}:
        return str(tool.invoke(arguments))
    if provider == "crewai":
        return str(tool.run(**arguments))
    if provider == "llamaindex":
        return str(tool.call(**arguments).content)
    if provider == "openai-agents":
        return str(asyncio.run(tool.on_invoke_tool(None, json.dumps(arguments))))
    if provider == "claude-agent-sdk":
        result = asyncio.run(tool.handler(arguments))
        return str(result["content"][0]["text"])
    if provider == "pydantic-ai":
        return str(tool.function(**arguments))
    if provider == "autogen":
        from autogen_core import CancellationToken

        return str(asyncio.run(tool.run_json(arguments, CancellationToken())))
    if provider == "microsoft-agent-framework":
        return str(tool(**arguments))
    if provider == "haystack":
        return str(tool.invoke(**arguments))
    if provider == "semantic-kernel":
        return str(tool(**arguments))
    if provider == "agno":
        return str(tool.entrypoint(**arguments))
    if provider == "smolagents":
        return str(tool.forward(**arguments))
    if provider == "google-adk":
        return str(tool.func(**arguments))
    raise AssertionError(f"Missing native invocation for {provider}")
