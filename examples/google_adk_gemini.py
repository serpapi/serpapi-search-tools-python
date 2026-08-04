# Run: uv run --isolated --no-project --with 'serpapi-search-tools[google-adk]' --with python-dotenv examples/google_adk_gemini.py  # noqa: E501
from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from serpapi_search_tools import shopping_search, web_search

load_dotenv()

MODEL = os.getenv("GOOGLE_ADK_MODEL", "gemini-flash-lite-latest")
PROMPT = (
    "Use the SerpApi tool to search Google Shopping for 'best travel backpacks'. "
    "Return three concise buying notes."
)
MAX_ATTEMPTS = 3


def _require_serpapi_key() -> None:
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")


def _configure_gemini_key() -> None:
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this example.")
    os.environ["GOOGLE_API_KEY"] = gemini_key
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


async def main() -> None:
    _require_serpapi_key()
    _configure_gemini_key()

    agent = Agent(
        name="serpapi_google_adk_agent",
        model=MODEL,
        instruction="Use the SerpApi tool when live web search helps.",
        generate_content_config=types.GenerateContentConfig(temperature=0),
        tools=[
            shopping_search(
                provider="google-adk",
                default_params={"num": "3", "hl": "en", "gl": "us"},
                include_examples=False,
            ),
            web_search(
                provider="google-adk",
                allowed_engines=["google_light"],
                include_examples=False,
            ),
        ],
    )
    app_name = "serpapi-search-tools-example"
    user_id = "example-user"
    session_id = "example-session"
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
            print("\n".join(final_text))
            return
        if attempt == MAX_ATTEMPTS:
            raise RuntimeError("Google ADK completed without a final text response.")


def _is_transient_gemini_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "503" in message or "unavailable" in message or "high demand" in message


if __name__ == "__main__":
    asyncio.run(main())
