# Run: uv run --with . --with semantic-kernel --with openai cookbook/semantic-kernel/main.py
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Build a competitor brief for US residential solar financing platforms. First "
        "create a short research plan, then use web search for company and product facts "
        "and news search for recent developments. Compare three competitors, identify "
        "evidence gaps, and include source URLs."
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
    path = output_dir / "semantic-kernel-competitor-brief.md"
    path.write_text(text)
    return path


async def main() -> None:
    _require_serpapi_key()
    kernel = Kernel()
    plugin = kernel.add_functions(
        "serpapi",
        [
            web_search(
                provider="semantic-kernel",
                allowed_engines=["google_light", "bing"],
                default_params={"num": "5", "hl": "en", "gl": "us"},
            ),
            news_search(
                provider="semantic-kernel",
                default_params={"num": "5", "hl": "en", "gl": "us"},
            ),
        ],
    )
    agent = ChatCompletionAgent(
        name="competitor_research_agent",
        instructions=(
            "Follow a plan-and-execute process: state a short plan, search each open "
            "question, inspect evidence quality, and then write the final competitor brief. "
            "Do not invent company facts or omit source URLs."
        ),
        service=OpenAIChatCompletion(
            ai_model_id=MODEL,
            api_key=_require_env("OPENAI_API_KEY"),
        ),
        plugins=[plugin],
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )
    responses: list[str] = []
    async for response in agent.invoke(messages=PROMPT):
        responses.append(str(response))
    if not responses:
        raise RuntimeError("Semantic Kernel completed without a final report.")
    report = responses[-1]
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    asyncio.run(main())
