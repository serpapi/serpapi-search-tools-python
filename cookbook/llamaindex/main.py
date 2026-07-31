# Run: uv run --isolated --no-project --with 'serpapi-search-tools[llamaindex]' --with python-dotenv --with llama-index-llms-openai cookbook/llamaindex/main.py  # noqa: E501
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLMMetadata, MessageRole
from llama_index.llms.openai import OpenAI

from serpapi_search_tools import maps_search, travel_explore_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Starting from JFK, identify three promising destinations for a four-day remote-work "
        "trip. Use travel exploration for indicative flight options, web search for practical "
        "remote-work facts, and maps search for coworking access. Recommend one destination "
        "with evidence, tradeoffs, and source URLs."
    ),
)


class OpenAICompatible(OpenAI):
    @property
    def _tokenizer(self) -> Any | None:
        return None

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=400_000,
            num_output=-1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this cookbook.")
    return value


def _serpapi_api_key() -> str:
    value = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not value:
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this cookbook.")
    return value


def _write_report(text: str) -> Path:
    output_dir = Path(os.getenv("COOKBOOK_OUTPUT_DIR", "cookbook-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "llamaindex-destination-brief.md"
    path.write_text(text)
    return path


async def main() -> None:
    api_key = _serpapi_api_key()
    llm = OpenAICompatible(
        model=MODEL,
        api_key=_require_env("OPENAI_API_KEY"),
        temperature=0,
    )
    agent = FunctionAgent(
        tools=[
            travel_explore_search(provider="llamaindex", api_key=api_key),
            web_search(
                provider="llamaindex",
                allowed_engines=["google_light", "bing"],
                default_params={"num": "5", "hl": "en", "gl": "us"},
                api_key=api_key,
            ),
            maps_search(
                provider="llamaindex",
                default_params={"hl": "en"},
                api_key=api_key,
            ),
        ],
        llm=llm,
        initial_tool_choice="required",
        streaming=False,
        timeout=180,
        system_prompt=(
            "You are a destination research agent. Gather evidence from each relevant tool, "
            "make the comparison explicit, and include source URLs."
        ),
    )
    result = await agent.run(PROMPT)
    report = str(result)
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    asyncio.run(main())
