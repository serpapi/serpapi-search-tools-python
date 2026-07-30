from __future__ import annotations

import pytest
from optional_dependencies import import_optional
from provider_helpers import (
    PROVIDER_IMPORTS,
    SUPPORTED_AGENT_PROVIDERS,
    _google_adk_litellm_model,
    prompt_for_tool,
    run_agent_provider,
)

pytestmark = pytest.mark.integration

PROVIDER_CASES = {
    "openai-agents": (
        "hotels_search",
        {
            "query": "Kyoto hotels",
            "check_in_date": "2026-08-01",
            "check_out_date": "2026-08-04",
        },
        "google_hotels",
    ),
    "pydantic-ai": ("images_search", {"query": "latte art"}, "google_images"),
    "langchain": ("maps_search", {"query": "coffee", "location": "Austin, Texas"}, "google_maps"),
    "langgraph": ("news_search", {"query": "Python releases"}, "google_news"),
    "agno": (
        "flights_search",
        {
            "departure_id": "LAX",
            "arrival_id": "AUS",
            "outbound_date": "2026-08-01",
        },
        "google_flights",
    ),
    "smolagents": ("videos_search", {"query": "latte art"}, "youtube"),
    "crewai": ("shopping_search", {"query": "espresso machine", "engine": "amazon"}, "amazon"),
    "autogen": ("news_search", {"query": "Python releases"}, "google_news"),
    "microsoft-agent-framework": (
        "web_search",
        {"query": "Python releases", "engine": "google_light"},
        "google_light",
    ),
    "haystack": ("maps_search", {"query": "coffee", "location": "Austin, Texas"}, "google_maps"),
    "llamaindex": ("travel_explore_search", {"departure_id": "JFK"}, "google_travel_explore"),
    "google-adk": ("web_search", {"query": "coffee", "engine": "google_light"}, "google_light"),
}


@pytest.mark.parametrize("provider", SUPPORTED_AGENT_PROVIDERS)
def test_each_agent_sdk_invokes_a_dedicated_public_tool(
    provider: str,
    fake_openai_server,
    serpapi_client,
) -> None:
    import_optional(PROVIDER_IMPORTS[provider])
    tool_name, arguments, expected_engine = PROVIDER_CASES[provider]

    result = run_agent_provider(
        provider,
        base_url=f"{fake_openai_server.url}/v1",
        api_key="fake-key",
        model="gpt-4o-mini",
        prompt=prompt_for_tool(tool_name, arguments),
        tool_names=[tool_name],
        client=serpapi_client,
    )

    assert serpapi_client.calls, (
        f"{provider} did not invoke its tool; result={result!r}; "
        f"requests={fake_openai_server.requests!r}"
    )
    assert serpapi_client.calls[0]["engine"] == expected_engine
    if tool_name in {"flights_search", "travel_explore_search"}:
        assert "q" not in serpapi_client.calls[0]


def test_openai_agents_selects_structured_tool_from_composed_tool_list(
    fake_openai_server,
    serpapi_client,
) -> None:
    import_optional("agents")
    arguments = {
        "departure_id": "LAX",
        "arrival_id": "AUS",
        "outbound_date": "2026-08-01",
    }

    run_agent_provider(
        "openai-agents",
        base_url=f"{fake_openai_server.url}/v1",
        api_key="fake-key",
        model="gpt-4o-mini",
        prompt=prompt_for_tool("flights_search", arguments),
        tool_names=["web_search", "maps_search", "hotels_search", "flights_search"],
        client=serpapi_client,
    )

    assert serpapi_client.calls[0]["engine"] == "google_flights"
    assert "q" not in serpapi_client.calls[0]


def test_google_adk_litellm_model_uses_openai_provider_for_compatible_models() -> None:
    assert _google_adk_litellm_model("gpt-4o-mini") == "openai/gpt-4o-mini"
    assert _google_adk_litellm_model("openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert _google_adk_litellm_model("google/gemma-4-e2b") == "openai/google/gemma-4-e2b"
