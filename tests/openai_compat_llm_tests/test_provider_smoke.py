from __future__ import annotations

import os
from importlib import import_module

import pytest
from model_retry import run_until_tool_called
from provider_helpers import (
    PROVIDER_IMPORTS,
    SUPPORTED_AGENT_PROVIDERS,
    prompt_for_tool,
    run_agent_provider,
)

pytestmark = pytest.mark.openai_compat_llm


@pytest.mark.parametrize("provider", SUPPORTED_AGENT_PROVIDERS)
def test_openai_compatible_llm_invokes_web_tool_for_every_supported_provider(
    provider: str,
    openai_compat_llm_settings,
    fake_serpapi_client,
) -> None:
    required_provider = os.getenv("SERPAPI_SEARCH_TOOL_REQUIRED_PROVIDER")
    if required_provider is not None and provider != required_provider:
        pytest.skip(f"isolated environment requires {required_provider}")
    if required_provider == provider:
        import_module(PROVIDER_IMPORTS[provider])

    arguments = {"query": "coffee", "engine": "google_light"}
    run_until_tool_called(
        lambda: run_agent_provider(
            provider,
            base_url=openai_compat_llm_settings.base_url,
            api_key=openai_compat_llm_settings.api_key,
            model=openai_compat_llm_settings.model,
            prompt=prompt_for_tool("web_search", arguments),
            tool_names=["web_search"],
            client=fake_serpapi_client,
        ),
        fake_serpapi_client,
    )

    expected = {"engine": "google_light", "q": "coffee"}
    assert fake_serpapi_client.calls
    assert all(call == expected for call in fake_serpapi_client.calls)
