# Run: uv run --isolated --no-project --with 'serpapi-search-tools[openai-agents]' --with python-dotenv examples/openai_agents_openai.py  # noqa: E501
from __future__ import annotations

import asyncio
import os

from _logging_client import LoggingClient
from agents import Agent, Runner
from dotenv import load_dotenv

from serpapi_search_tools import maps_search, news_search, shopping_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = (
    "Research noise-cancelling headphones: compare three current shopping results, "
    "find recent news about the leading maker, and find a nearby electronics store "
    "in Austin. Use shopping, news, and maps search, then give a concise summary."
)


def _require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"Set {name} before running this example.")


def _require_serpapi_key() -> str:
    value = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not value:
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")
    return value


async def main() -> None:
    _require_env("OPENAI_API_KEY")
    client = LoggingClient(_require_serpapi_key())
    agent = Agent(
        name="serpapi-openai-agent",
        instructions="Use the most specific SerpApi tool for each part of the request.",
        model=MODEL,
        tools=[
            web_search(
                provider="openai-agents",
                allowed_engines=["google_light", "google"],
                default_params={"hl": "en", "gl": "us"},
                result_limit=3,
                client=client,
            ),
            news_search(provider="openai-agents", result_limit=3, client=client),
            maps_search(
                provider="openai-agents",
                default_params={"hl": "en", "gl": "us"},
                client=client,
            ),
            shopping_search(provider="openai-agents", result_limit=3, client=client),
        ],
    )
    result = await Runner.run(agent, PROMPT)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
