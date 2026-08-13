# Run: uv run --isolated --no-project --with 'serpapi-search-tools[smolagents]' --with python-dotenv examples/smolagents_openai.py  # noqa: E501
from __future__ import annotations

import os

from dotenv import load_dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent

from serpapi_search_tools import videos_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = (
    "Use the SerpApi tool to search YouTube for 'beginner pour over coffee tutorial'. "
    "Recommend one useful video angle."
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
    model = OpenAIServerModel(
        model_id=MODEL,
        api_key=_require_env("OPENAI_API_KEY"),
    )
    agent = ToolCallingAgent(
        model=model,
        tools=[
            videos_search(
                provider="smolagents",
                default_params={"hl": "en", "gl": "us"},
                result_limit=3,
                include_examples=False,
            )
        ],
        max_steps=3,
    )
    print(agent.run(PROMPT))


if __name__ == "__main__":
    main()
