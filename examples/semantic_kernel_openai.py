# Run: uv run --isolated --no-project --with 'serpapi-search-tools[semantic-kernel]' --with python-dotenv examples/semantic_kernel_openai.py  # noqa: E501
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from serpapi_search_tools import maps_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = "Find coffee roasters in Portland and use web search to compare two of them."


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example.")
    return value


async def main() -> None:
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")

    kernel = Kernel()
    plugin = kernel.add_functions(
        "serpapi",
        [
            maps_search(provider="semantic-kernel"),
            web_search(provider="semantic-kernel", allowed_engines=["google_light"]),
        ],
    )
    agent = ChatCompletionAgent(
        name="serpapi_search_agent",
        instructions="Use the SerpApi tools for current search data.",
        service=OpenAIChatCompletion(
            ai_model_id=MODEL,
            api_key=_require_env("OPENAI_API_KEY"),
        ),
        plugins=[plugin],
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )

    async for response in agent.invoke(messages=PROMPT):
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
