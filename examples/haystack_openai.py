# Run: uv run --isolated --no-project --with 'serpapi-search-tools[haystack]' --with python-dotenv examples/haystack_openai.py  # noqa: E501
from __future__ import annotations

import os

from dotenv import load_dotenv
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret

from serpapi_search_tools import maps_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = (
    "Use the SerpApi tool to search Google for 'best hiking trails near Boulder'. "
    "Give a compact weekend shortlist."
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example.")
    return value


def _require_serpapi_key() -> None:
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")


def main() -> None:
    _require_serpapi_key()
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(_require_env("OPENAI_API_KEY")),
        model=MODEL,
    )
    agent = Agent(
        chat_generator=generator,
        tools=[
            maps_search(
                provider="haystack",
                default_params={"hl": "en", "gl": "us"},
                result_limit=3,
            ),
            web_search(provider="haystack", allowed_engines=["google_light"]),
        ],
        max_agent_steps=3,
    )
    result = agent.run(messages=[ChatMessage.from_user(PROMPT)])
    print(result)


if __name__ == "__main__":
    main()
