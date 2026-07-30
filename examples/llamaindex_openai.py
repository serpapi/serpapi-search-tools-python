# Run: uv run --with '.[llamaindex]' --with llama-index-llms-openai examples/llamaindex_openai.py
from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLMMetadata, MessageRole
from llama_index.llms.openai import OpenAI

from serpapi_search_tools import travel_explore_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = "Explore destinations from JFK, then research the best city for remote workers."


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
        raise RuntimeError(f"Set {name} before running this example.")
    return value


def _serpapi_api_key() -> str:
    value = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not value:
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")
    return value


async def main() -> None:
    llm = OpenAICompatible(
        model=MODEL,
        api_key=_require_env("OPENAI_API_KEY"),
        temperature=0,
    )
    agent = FunctionAgent(
        tools=[
            travel_explore_search(
                provider="llamaindex",
                api_key=_serpapi_api_key(),
            ),
            web_search(
                provider="llamaindex",
                allowed_engines=["google_light"],
                default_params={"num": "3", "hl": "en", "gl": "us"},
                api_key=_serpapi_api_key(),
            ),
        ],
        llm=llm,
        initial_tool_choice="required",
        streaming=False,
        timeout=120,
    )
    result = await agent.run(PROMPT)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
