import os
from importlib import import_module
from types import ModuleType

import pytest

_PROVIDER_MODULE_PREFIXES: dict[str, tuple[str, ...]] = {
    "agno": ("agno",),
    "autogen": ("autogen_agentchat", "autogen_core", "autogen_ext"),
    "claude-agent-sdk": ("claude_agent_sdk",),
    "crewai": ("crewai",),
    "google-adk": ("google.adk",),
    "haystack": ("haystack",),
    "langchain": ("langchain", "langchain_openai"),
    "langgraph": ("langchain", "langchain_openai", "langgraph"),
    "llamaindex": ("llama_index",),
    "openai-agents": ("agents",),
    "pydantic-ai": ("pydantic_ai",),
    "semantic-kernel": ("semantic_kernel",),
    "smolagents": ("smolagents",),
}


def import_optional(module_name: str) -> ModuleType:
    """Import an optional package without hiding failures inside that package."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name
        if missing_name == module_name or (
            missing_name is not None and module_name.startswith(f"{missing_name}.")
        ):
            required_provider = os.getenv("SERPAPI_SEARCH_TOOL_REQUIRED_PROVIDER")
            required_prefixes = _PROVIDER_MODULE_PREFIXES.get(required_provider or "", ())
            if any(
                module_name == prefix or module_name.startswith(f"{prefix}.")
                for prefix in required_prefixes
            ):
                pytest.fail(
                    f"required provider {required_provider!r} is missing module {module_name!r}",
                    pytrace=False,
                )
            pytest.skip(f"optional dependency {module_name!r} is not installed")
        raise
