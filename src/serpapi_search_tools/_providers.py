from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Literal, TypeAlias

ProviderName: TypeAlias = Literal[
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
]


@dataclass(frozen=True)
class ProviderSpec:
    """Metadata needed to register and auto-detect one agent SDK adapter."""

    name: str
    aliases: tuple[str, ...]
    distribution: str | None
    extra: str | None


PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec("langgraph", (), "langgraph", "langgraph"),
    ProviderSpec("langchain", (), "langchain-core", "langchain"),
    ProviderSpec("crewai", (), "crewai", "crewai"),
    ProviderSpec("llamaindex", ("llama_index", "llama-index"), "llama-index-core", "llamaindex"),
    ProviderSpec(
        "openai-agents",
        ("openai_agents",),
        "openai-agents",
        "openai-agents",
    ),
    ProviderSpec(
        "claude-agent-sdk",
        (
            "claude_agent_sdk",
            "claude-sdk",
            "claude_agent",
            "anthropic-agent-sdk",
            "anthropic_agent_sdk",
            "anthropic-agents",
            "anthropic_agents",
        ),
        "claude-agent-sdk",
        "claude-agent-sdk",
    ),
    ProviderSpec("pydantic-ai", ("pydantic_ai",), "pydantic-ai", "pydantic-ai"),
    ProviderSpec("autogen", (), "autogen-core", "autogen"),
    ProviderSpec("haystack", (), "haystack-ai", "haystack"),
    ProviderSpec(
        "semantic-kernel",
        ("semantic_kernel",),
        "semantic-kernel",
        "semantic-kernel",
    ),
    ProviderSpec("agno", (), "agno", "agno"),
    ProviderSpec("smolagents", ("smol-agents",), "smolagents", "smolagents"),
    ProviderSpec("google-adk", ("google_adk", "adk"), "google-adk", "google-adk"),
    ProviderSpec("function", ("plain", "callable"), None, None),
)

PROVIDER_ALIASES: dict[str, str] = {"auto": "auto"}
for _spec in PROVIDER_SPECS:
    PROVIDER_ALIASES[_spec.name] = _spec.name
    PROVIDER_ALIASES.update(dict.fromkeys(_spec.aliases, _spec.name))


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[normalized]
    return PROVIDER_ALIASES.get(normalized.replace("_", "-"), normalized)


def detect_provider() -> str:
    for spec in PROVIDER_SPECS:
        if spec.distribution is None:
            continue
        try:
            importlib.metadata.version(spec.distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        return spec.name
    return "function"
