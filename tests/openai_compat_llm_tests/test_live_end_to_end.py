from __future__ import annotations

import pytest
from model_retry import run_until_tool_called
from provider_helpers import prompt_for_tool, run_agent_provider

pytestmark = pytest.mark.openai_compat_llm


def test_openai_compatible_llm_to_agent_to_live_serpapi(
    openai_compat_llm_settings,
    live_serpapi_client,
) -> None:
    run_until_tool_called(
        lambda: run_agent_provider(
            "openai-agents",
            base_url=openai_compat_llm_settings.base_url,
            api_key=openai_compat_llm_settings.api_key,
            model=openai_compat_llm_settings.model,
            prompt=prompt_for_tool("web_search", {"query": "SerpApi", "engine": "google_light"}),
            tool_names=["web_search"],
            client=live_serpapi_client,
        ),
        live_serpapi_client,
    )

    assert live_serpapi_client.calls == [{"engine": "google_light", "q": "SerpApi"}]
