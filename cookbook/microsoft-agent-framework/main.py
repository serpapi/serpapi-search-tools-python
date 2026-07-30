# Run: uv run --with '.[microsoft-agent-framework]' cookbook/microsoft-agent-framework/main.py
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Prepare a technology due-diligence memo on warehouse robotics. Find five recent "
        "developments with news search, then use web search to verify the strongest "
        "company claims against durable sources. Separate verified facts, vendor claims, "
        "and analyst inference. Finish with adoption risks and direct source URLs."
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
    path = output_dir / "microsoft-agent-framework-due-diligence.md"
    path.write_text(text)
    return path


async def main() -> None:
    _require_serpapi_key()
    agent = Agent(
        client=OpenAIChatClient(
            model=MODEL,
            api_key=_require_env("OPENAI_API_KEY"),
        ),
        name="technology_due_diligence_agent",
        instructions=(
            "Act as a careful technology analyst. Use live SerpApi tools, cross-check "
            "material claims, label uncertainty, and include direct source URLs."
        ),
        tools=[
            news_search(
                provider="microsoft-agent-framework",
                default_params={"num": "8", "hl": "en", "gl": "us"},
            ),
            web_search(
                provider="microsoft-agent-framework",
                allowed_engines=["google_light", "bing"],
                default_params={"num": "6", "hl": "en", "gl": "us"},
            ),
        ],
    )
    report = str(await agent.run(PROMPT))
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    asyncio.run(main())
