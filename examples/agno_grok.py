# Run: uv run --isolated --no-project --with 'serpapi-search-tools[agno]' --with python-dotenv examples/agno_grok.py  # noqa: E501
from __future__ import annotations

import os

from agno.agent import Agent
from agno.models.xai import xAI
from dotenv import load_dotenv

from serpapi_search_tools import shopping_search, web_search

load_dotenv()

MODEL = os.getenv("XAI_MODEL", "grok-4.5")
BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
PROMPT = (
    "Use the SerpApi tool to search Google for 'best espresso grinder'. "
    "Summarize the useful buying signals."
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
    api_key = _require_env("XAI_API_KEY")
    model = xAI(
        id=MODEL,
        api_key=api_key,
        base_url=BASE_URL,
    )
    agent = Agent(
        model=model,
        instructions="Use the SerpApi tool whenever live search helps.",
        tools=[
            web_search(
                provider="agno",
                allowed_engines=["google_light", "google"],
                default_params={"hl": "en", "gl": "us"},
                result_limit=3,
                include_examples=False,
            ),
            shopping_search(
                provider="agno",
                result_limit=3,
                include_examples=False,
            ),
        ],
        markdown=True,
        telemetry=False,
    )
    agent.print_response(PROMPT)


if __name__ == "__main__":
    main()
