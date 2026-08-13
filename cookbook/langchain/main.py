# Run: uv run --isolated --no-project --with 'serpapi-search-tools[langchain]' --with python-dotenv --with langchain-openai cookbook/langchain/main.py  # noqa: E501
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("XAI_MODEL", "grok-4.5")
BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Prepare a decision-ready market brief on commercial battery recycling in the US. "
        "Plan the research, use web search for durable sources and news search for recent "
        "developments, compare at least three independent sources, and finish with market "
        "signals, risks, and unanswered questions. Include source URLs."
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
    path = output_dir / "langchain-market-research.md"
    path.write_text(text)
    return path


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
            web_search(
                provider="langchain",
                allowed_engines=["google_light", "bing"],
                result_limit=5,
                include_examples=False,
            ),
            news_search(
                provider="langchain",
                default_params={"hl": "en", "gl": "us"},
                result_limit=5,
                include_examples=False,
            ),
        ],
        system_prompt=(
            "You are a careful research agent. Make a short plan before searching, "
            "distinguish current reporting from durable background sources, and never "
            "present an unsupported claim as fact."
        ),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": PROMPT}]})
    report = str(result["messages"][-1].content)
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    main()
