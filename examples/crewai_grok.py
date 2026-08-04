# Run: uv run --isolated --no-project --with 'serpapi-search-tools[crewai]' --with python-dotenv examples/crewai_grok.py  # noqa: E501
from __future__ import annotations

import os

import crewai
from dotenv import load_dotenv

from serpapi_search_tools import shopping_search

load_dotenv()

MODEL = os.getenv("XAI_MODEL", "grok-4.5")
BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
PROMPT = (
    "Use the SerpApi tool to search Google Shopping for 'noise cancelling headphones'. "
    "Compare the top options briefly."
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
    llm = crewai.LLM(
        model=MODEL,
        api_key=_require_env("XAI_API_KEY"),
        base_url=BASE_URL,
        provider="openai",
    )
    agent = crewai.Agent(
        role="Shopping search analyst",
        goal="Use SerpApi search results to compare products.",
        backstory="You prefer fresh search data over memory.",
        llm=llm,
        tools=[
            shopping_search(
                provider="crewai",
                default_params={"num": "3", "hl": "en", "gl": "us"},
            )
        ],
        max_iter=3,
        verbose=False,
    )
    task = crewai.Task(description=PROMPT, expected_output="A concise comparison.", agent=agent)
    crew = crewai.Crew(agents=[agent], tasks=[task], verbose=False)
    print(crew.kickoff())


if __name__ == "__main__":
    main()
