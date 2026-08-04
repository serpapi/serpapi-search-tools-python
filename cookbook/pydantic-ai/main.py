# Run: uv run --isolated --no-project --with 'serpapi-search-tools[pydantic-ai]' --with python-dotenv cookbook/pydantic-ai/main.py  # noqa: E501
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from serpapi_search_tools import images_search, maps_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Scout Portland, Oregon for an outdoor lifestyle campaign. Find three visually "
        "distinct locations using image search, verify access and nearby landmarks with "
        "maps, and use web search for restrictions or practical context. Return a location "
        "brief with visual motifs, logistics, risks, and source URLs."
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
    path = output_dir / "pydantic-ai-location-scout.md"
    path.write_text(text)
    return path


def main() -> None:
    _require_serpapi_key()
    provider = OpenAIProvider(api_key=_require_env("OPENAI_API_KEY"))
    model = OpenAIResponsesModel(MODEL, provider=provider)
    agent = Agent(
        model,
        instructions=(
            "You are a location scout. Use image search for visual evidence, maps for "
            "place facts, and web search for practical verification. Do not infer access "
            "or permissions from an image alone."
        ),
        tools=[
            images_search(
                provider="pydantic-ai",
                default_params={"num": "5", "hl": "en", "gl": "us", "safe": "active"},
            ),
            maps_search(
                provider="pydantic-ai",
                default_params={"hl": "en", "gl": "us"},
            ),
            web_search(
                provider="pydantic-ai",
                allowed_engines=["google_light", "bing"],
                default_params={"num": "5", "hl": "en", "gl": "us"},
            ),
        ],
    )
    result = agent.run_sync(PROMPT)
    report = str(getattr(result, "output", result))
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    main()
