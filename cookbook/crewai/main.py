# Run: uv run --isolated --no-project --with 'serpapi-search-tools[crewai]' --with python-dotenv cookbook/crewai/main.py  # noqa: E501
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import crewai
from dotenv import load_dotenv

from serpapi_search_tools import flights_search, hotels_search, maps_search

load_dotenv()

MODEL = os.getenv("XAI_MODEL", "grok-4.5")
BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")


def _default_prompt() -> str:
    departure = date.today() + timedelta(days=120)
    return_date = departure + timedelta(days=4)
    return (
        "Plan a four-night Kyoto trip for two adults. Fly SFO to KIX on "
        f"{departure.isoformat()} and return {return_date.isoformat()}. Keep flights and "
        "lodging below USD 4,000, prefer a walkable neighborhood with easy transit, and "
        "include three nearby places worth visiting. Use current search evidence and "
        "state where prices may change."
    )


PROMPT = os.getenv("COOKBOOK_PROMPT", _default_prompt())


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this cookbook.")
    return value


def _require_serpapi_key() -> str:
    value = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not value:
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this cookbook.")
    return value


def _write_report(text: str) -> Path:
    output_dir = Path(os.getenv("COOKBOOK_OUTPUT_DIR", "cookbook-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "crewai-trip-plan.md"
    path.write_text(text)
    return path


def main() -> None:
    _require_serpapi_key()
    llm = crewai.LLM(
        model=MODEL,
        api_key=_require_env("XAI_API_KEY"),
        base_url=BASE_URL,
        provider="openai",
        timeout=90,
        max_retries=2,
    )
    travel_tools = [
        flights_search(
            provider="crewai",
            default_params={"currency": "USD", "hl": "en"},
        ),
        hotels_search(
            provider="crewai",
            default_params={"currency": "USD", "gl": "us", "hl": "en"},
        ),
        maps_search(
            provider="crewai",
            default_params={"hl": "en", "gl": "jp"},
        ),
    ]
    researcher = crewai.Agent(
        role="Travel search researcher",
        goal="Collect current flight, hotel, and neighborhood evidence for a concrete trip.",
        backstory="You verify live options before making a recommendation.",
        llm=llm,
        tools=travel_tools,
        max_iter=7,
        max_execution_time=180,
        verbose=False,
    )
    planner = crewai.Agent(
        role="Travel decision editor",
        goal="Turn verified options into a practical, budget-aware trip decision.",
        backstory="You expose uncertainty and never invent prices or availability.",
        llm=llm,
        max_iter=3,
        max_execution_time=180,
        verbose=False,
    )
    research_task = crewai.Task(
        description=PROMPT,
        expected_output=(
            "A research packet with flight options, hotel options, nearby places, prices, "
            "and source URLs."
        ),
        agent=researcher,
    )
    planning_task = crewai.Task(
        description=(
            "Use the research packet to recommend one flight and one hotel, show the "
            "estimated total, provide a compact four-day outline, and identify assumptions."
        ),
        expected_output="A decision-ready Markdown trip plan with source URLs.",
        agent=planner,
        context=[research_task],
    )
    crew = crewai.Crew(
        agents=[researcher, planner],
        tasks=[research_task, planning_task],
        process=crewai.Process.sequential,
        verbose=False,
    )
    report = str(crew.kickoff())
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    main()
