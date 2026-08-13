# Run: uv run --isolated --no-project --with 'serpapi-search-tools[agno]' --with python-dotenv cookbook/agno/main.py  # noqa: E501
from __future__ import annotations

import os
from pathlib import Path

from agno.agent import Agent
from agno.models.xai import xAI
from dotenv import load_dotenv

from serpapi_search_tools import news_search, shopping_search, web_search

load_dotenv()

MODEL = os.getenv("XAI_MODEL", "grok-4.5")
BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Assess the US market for home espresso grinders below USD 800. Use web search for "
        "category and manufacturer facts, news search for recent launches, and shopping "
        "search for live price and availability signals. Produce a market report covering "
        "segments, representative products, gaps, risks, and source URLs."
    ),
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this cookbook.")
    return value


def _require_serpapi_key() -> None:
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this cookbook.")


def _write_report(text: str) -> Path:
    output_dir = Path(os.getenv("COOKBOOK_OUTPUT_DIR", "cookbook-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "agno-market-research.md"
    path.write_text(text)
    return path


def main() -> None:
    _require_serpapi_key()
    model = xAI(
        id=MODEL,
        api_key=_require_env("XAI_API_KEY"),
        base_url=BASE_URL,
    )
    agent = Agent(
        name="SerpApi market researcher",
        model=model,
        instructions=[
            "Search before making market claims.",
            "Use the most specific SerpApi tool for each question.",
            "Separate live listing signals from durable category facts.",
            "Include source URLs and call out missing evidence.",
        ],
        tools=[
            web_search(
                provider="agno",
                allowed_engines=["google_light", "bing"],
                result_limit=5,
                include_examples=False,
            ),
            news_search(
                provider="agno",
                default_params={"hl": "en", "gl": "us"},
                result_limit=5,
                include_examples=False,
            ),
            shopping_search(
                provider="agno",
                result_limit=5,
                include_examples=False,
            ),
        ],
        markdown=True,
        telemetry=False,
        tool_call_limit=10,
    )
    response = agent.run(PROMPT)
    report = str(response.content)
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    main()
