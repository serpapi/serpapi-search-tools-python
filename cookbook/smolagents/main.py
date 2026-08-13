# Run: uv run --isolated --no-project --with 'serpapi-search-tools[smolagents]' --with python-dotenv cookbook/smolagents/main.py  # noqa: E501
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent

from serpapi_search_tools import images_search, shopping_search, videos_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Help a beginner choose a compact espresso machine below USD 700. Use shopping "
        "search for live listings, image search to compare counter footprint and controls, "
        "and video search for setup or maintenance demonstrations. Recommend up to three "
        "options with tradeoffs, evidence limitations, and source URLs."
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
    path = output_dir / "smolagents-purchase-research.md"
    path.write_text(text)
    return path


def main() -> None:
    _require_serpapi_key()
    model = OpenAIServerModel(
        model_id=MODEL,
        api_key=_require_env("OPENAI_API_KEY"),
    )
    agent = ToolCallingAgent(
        model=model,
        tools=[
            shopping_search(
                provider="smolagents",
                result_limit=5,
                include_examples=False,
            ),
            images_search(
                provider="smolagents",
                default_params={"hl": "en", "gl": "us", "safe": "active"},
                result_limit=5,
                include_examples=False,
            ),
            videos_search(
                provider="smolagents",
                default_params={"hl": "en", "gl": "us"},
                result_limit=5,
                include_examples=False,
            ),
        ],
        instructions=(
            "Use all relevant tools, treat listing prices as time-sensitive, and do not "
            "infer physical dimensions from an image without corroboration."
        ),
        max_steps=8,
    )
    report = str(agent.run(PROMPT))
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    main()
