# Run: uv run --with '.[autogen]' cookbook/autogen/main.py
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Research three companies competing in residential energy-management software. "
        "Use web search for company facts and news search for current moves. Produce a "
        "company intelligence memo comparing positioning, recent signals, risks, and "
        "evidence quality. Include source URLs and do not invent financial figures."
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
    path = output_dir / "autogen-company-intelligence.md"
    path.write_text(text)
    return path


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
            "company_researcher",
            model_client=model_client,
            system_message=(
                "You are a company research analyst. Search iteratively, distinguish "
                "company claims from independent reporting, and cite source URLs."
            ),
            tools=[
                web_search(
                    provider="autogen",
                    allowed_engines=["google_light", "bing"],
                    default_params={"num": "5", "hl": "en", "gl": "us"},
                ),
                news_search(
                    provider="autogen",
                    default_params={"num": "5", "hl": "en", "gl": "us"},
                ),
            ],
            reflect_on_tool_use=True,
            max_tool_iterations=8,
        )
        result = await agent.run(task=PROMPT)
        last_message = result.messages[-1]
        report = str(last_message.content)
    finally:
        await model_client.close()
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    asyncio.run(main())
