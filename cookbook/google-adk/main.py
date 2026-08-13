# Run: uv run --isolated --no-project --with 'serpapi-search-tools[google-adk]' --with python-dotenv cookbook/google-adk/main.py  # noqa: E501
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from serpapi_search_tools import maps_search, news_search, web_search

load_dotenv()

MODEL = os.getenv("GOOGLE_ADK_MODEL", "gemini-flash-lite-latest")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Recommend one of Austin, Raleigh, or Denver for the first location of a specialty "
        "coffee roaster. Use maps search to inspect competitor and neighborhood patterns, "
        "web search for durable city context, and news search for recent local developments. "
        "Produce a location strategy memo with evidence, tradeoffs, and source URLs."
    ),
)
MAX_ATTEMPTS = 3


def _require_serpapi_key() -> str:
    value = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not value:
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this cookbook.")
    return value


def _configure_gemini_key() -> None:
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this cookbook.")
    os.environ["GOOGLE_API_KEY"] = gemini_key
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


def _write_report(text: str) -> Path:
    output_dir = Path(os.getenv("COOKBOOK_OUTPUT_DIR", "cookbook-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "google-adk-location-strategy.md"
    path.write_text(text)
    return path


async def main() -> None:
    _require_serpapi_key()
    _configure_gemini_key()
    agent = Agent(
        name="serpapi_retail_location_strategist",
        model=MODEL,
        instruction=(
            "Act as a retail location strategist. Use maps for place-level evidence, web "
            "for durable context, and news for current signals. Compare cities consistently "
            "and include direct source URLs."
        ),
        generate_content_config=types.GenerateContentConfig(temperature=0),
        tools=[
            maps_search(
                provider="google-adk",
                default_params={"hl": "en", "gl": "us"},
                include_examples=False,
            ),
            web_search(
                provider="google-adk",
                allowed_engines=["google_light", "bing"],
                result_limit=5,
                include_examples=False,
            ),
            news_search(
                provider="google-adk",
                default_params={"hl": "en", "gl": "us"},
                result_limit=5,
                include_examples=False,
            ),
        ],
    )
    app_name = "serpapi-retail-location-cookbook"
    user_id = "cookbook-user"
    session_id = "location-strategy-session"
    session_service = InMemorySessionService()
    await _maybe_await(
        session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
    )
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    message = types.Content(role="user", parts=[types.Part(text=PROMPT)])

    for attempt in range(1, MAX_ATTEMPTS + 1):
        final_text: list[str] = []
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message,
            ):
                if event.is_final_response() and event.content:
                    for part in event.content.parts or []:
                        if part.text:
                            final_text.append(part.text)
        except Exception as exc:
            if attempt == MAX_ATTEMPTS or not _is_transient_gemini_error(exc):
                raise
            await asyncio.sleep(2**attempt)
            continue

        if final_text:
            report = "\n".join(final_text)
            path = _write_report(report)
            print(report)
            print(f"\nSaved report to {path}")
            return
        if attempt == MAX_ATTEMPTS:
            raise RuntimeError("Google ADK completed without a final report.")


def _is_transient_gemini_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "503" in message or "unavailable" in message or "high demand" in message


if __name__ == "__main__":
    asyncio.run(main())
