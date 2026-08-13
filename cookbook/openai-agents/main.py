# Run: uv run --isolated --no-project --with 'serpapi-search-tools[openai-agents]' --with python-dotenv cookbook/openai-agents/main.py  # noqa: E501
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agents import Agent, Runner
from dotenv import load_dotenv

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Create a research report on the near-term outlook for heat-pump adoption in the "
        "United States. Cover policy, consumer economics, and manufacturer activity. "
        "Use current evidence, cite source URLs, identify disagreements, and finish with "
        "three signals to monitor over the next year."
    ),
)


def _require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"Set {name} before running this cookbook.")


def _require_serpapi_key() -> None:
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this cookbook.")


def _write_report(text: str) -> Path:
    output_dir = Path(os.getenv("COOKBOOK_OUTPUT_DIR", "cookbook-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "openai-agents-research-report.md"
    path.write_text(text)
    return path


async def main() -> None:
    _require_serpapi_key()
    _require_env("OPENAI_API_KEY")
    researcher = Agent(
        name="SerpApi research specialist",
        instructions=(
            "Use web search for durable sources and news search for recent reporting. "
            "Return compact research notes with URLs and label uncertainty."
        ),
        model=MODEL,
        tools=[
            web_search(
                provider="openai-agents",
                allowed_engines=["google_light", "bing"],
                result_limit=5,
            ),
            news_search(
                provider="openai-agents",
                default_params={"hl": "en", "gl": "us"},
                result_limit=5,
            ),
        ],
    )
    editor = Agent(
        name="Research report editor",
        instructions=(
            "Own the final report. Delegate searches to the research specialist, ask "
            "focused follow-up questions when evidence is missing, and produce a concise "
            "Markdown report that separates facts, analysis, and open questions."
        ),
        model=MODEL,
        tools=[
            researcher.as_tool(
                tool_name="research_with_serpapi",
                tool_description=(
                    "Research a focused question using live SerpApi web and news data."
                ),
            )
        ],
    )
    result = await Runner.run(editor, PROMPT, max_turns=10)
    report = str(result.final_output)
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    asyncio.run(main())
