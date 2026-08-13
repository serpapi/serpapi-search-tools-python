# Run: uv run --isolated --no-project --with 'serpapi-search-tools[microsoft-agent-framework]' --with python-dotenv examples/microsoft_agent_framework_openai.py  # noqa: E501
from __future__ import annotations

import asyncio
import os

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = (
    "Research the latest warehouse-robotics developments. Use news search for recent "
    "events and web search to verify the two most important company claims. Summarize "
    "the findings with source URLs."
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example.")
    return value


async def main() -> None:
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")

    agent = Agent(
        client=OpenAIChatClient(
            model=MODEL,
            api_key=_require_env("OPENAI_API_KEY"),
        ),
        name="search_agent",
        instructions=(
            "Use SerpApi for current information. Distinguish reporting from company "
            "claims and include direct source URLs."
        ),
        tools=[
            news_search(
                provider="microsoft-agent-framework",
                default_params={"hl": "en", "gl": "us"},
                result_limit=5,
            ),
            web_search(
                provider="microsoft-agent-framework",
                allowed_engines=["google_light", "bing"],
                result_limit=5,
            ),
        ],
    )
    print(await agent.run(PROMPT))


if __name__ == "__main__":
    asyncio.run(main())
