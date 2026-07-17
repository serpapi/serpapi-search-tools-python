# Run: uv run --with '.[autogen]' examples/autogen_openai.py
from __future__ import annotations

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = (
    "Use the SerpApi tool to search Google News for 'electric vehicle battery recycling'. "
    "Then give a short trend summary."
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example.")
    return value


def _require_serpapi_key() -> None:
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")


async def main() -> None:
    _require_serpapi_key()
    model_client = OpenAIChatCompletionClient(
        model=MODEL,
        api_key=_require_env("OPENAI_API_KEY"),
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )
    try:
        agent = AssistantAgent(
            "search_agent",
            model_client=model_client,
            tools=[
                news_search(
                    provider="autogen",
                    default_params={"num": "3", "hl": "en", "gl": "us"},
                ),
                web_search(provider="autogen", allowed_engines=["google_light"]),
            ],
            reflect_on_tool_use=True,
        )
        result = await agent.run(task=PROMPT)
        print(result)
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
