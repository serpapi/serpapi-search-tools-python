# Run: uv run --with . --with pydantic-ai examples/pydantic_ai_openai.py
from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from serpapi_search_tools import images_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = (
    "Use the SerpApi tool to search Google Images for 'minimal desk setup'. "
    "Describe the visual themes."
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
    provider = OpenAIProvider(api_key=_require_env("OPENAI_API_KEY"))
    model = OpenAIChatModel(MODEL, provider=provider)
    agent = Agent(
        model,
        instructions="Use the SerpApi tool for live image search.",
        tools=[
            images_search(
                provider="pydantic-ai",
                default_params={"num": "3", "hl": "en", "gl": "us"},
            ),
            web_search(provider="pydantic-ai", allowed_engines=["google_light"]),
        ],
    )
    result = agent.run_sync(PROMPT)
    print(getattr(result, "output", result))


if __name__ == "__main__":
    main()
