# Run: uv run --with . --with openai-agents examples/openai_agents_travel_planner.py
"""Give an OpenAI Agents planner the three typed travel search tools."""

from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta

from _logging_client import LoggingClient
from agents import Agent, Runner
from dotenv import load_dotenv

from serpapi_search_tools import flights_search, hotels_search, travel_explore_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


def _require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"Set {name} before running this example.")


def _require_serpapi_key() -> str:
    value = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not value:
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")
    return value


async def main() -> None:
    _require_env("OPENAI_API_KEY")
    client = LoggingClient(_require_serpapi_key(), max_results=3)
    outbound = date.today() + timedelta(days=60)
    returning = outbound + timedelta(days=4)
    prompt = (
        "Plan an Austin trip from LAX. First use travel explore with only departure_id='LAX' "
        "and omit every optional field so I can see flexible destination ideas. Then find a "
        f"round-trip business-class flight for one adult from {outbound.isoformat()} to "
        f"{returning.isoformat()}, and finally find an Austin hotel for those dates for "
        "two adults and one seven-year-old child. Use all three travel tools."
    )
    agent = Agent(
        name="serpapi-travel-planner",
        instructions=(
            "Use each typed SerpApi travel tool requested by the user. For travel explore, "
            "send only departure_id and omit destination, date, cabin, and passenger fields."
        ),
        model=MODEL,
        tools=[
            travel_explore_search(provider="openai-agents", client=client),
            flights_search(provider="openai-agents", client=client),
            hotels_search(provider="openai-agents", client=client),
        ],
    )
    result = await Runner.run(agent, prompt, max_turns=8)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
