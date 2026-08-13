# Run: uv run --isolated --no-project --with 'serpapi-search-tools[langchain]' --with python-dotenv --with langchain-openai examples/langchain_grok.py  # noqa: E501
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from serpapi_search_tools import maps_search, web_search

load_dotenv()

MODEL = os.getenv("XAI_MODEL", "grok-4.5")
BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
PROMPT = (
    "Use the SerpApi tool to search Google Maps for 'coffee roasters in Portland'. "
    "Return a short shortlist."
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example.")
    return value


def _require_serpapi_key() -> None:
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")


def main() -> None:
    _require_serpapi_key()
    model = ChatOpenAI(
        model=MODEL,
        api_key=_require_env("XAI_API_KEY"),
        base_url=BASE_URL,
        temperature=0,
    )
    agent = create_agent(
        model=model,
        tools=[
            maps_search(
                provider="langchain",
                default_params={"hl": "en", "gl": "us"},
                result_limit=3,
                include_examples=False,
            ),
            web_search(
                provider="langchain",
                allowed_engines=["google_light"],
                include_examples=False,
            ),
        ],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": PROMPT}]})
    print(result)


if __name__ == "__main__":
    main()
