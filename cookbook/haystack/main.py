# Run: uv run --isolated --no-project --with 'serpapi-search-tools[haystack]' --with python-dotenv cookbook/haystack/main.py  # noqa: E501
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Create this week's concise newsletter on warehouse robotics. Use news search to "
        "find recent developments and web search to verify background claims. Include a "
        "lead story, three briefs, why each matters, and source URLs. Avoid repeating "
        "syndicated versions of the same story."
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


def _report_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if messages:
        message = messages[-1]
        text = getattr(message, "text", None)
        if text:
            return str(text)
        return str(message)
    return str(result)


def _write_report(text: str) -> Path:
    output_dir = Path(os.getenv("COOKBOOK_OUTPUT_DIR", "cookbook-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "haystack-industry-newsletter.md"
    path.write_text(text)
    return path


def main() -> None:
    _require_serpapi_key()
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(_require_env("OPENAI_API_KEY")),
        model=MODEL,
    )
    agent = Agent(
        chat_generator=generator,
        system_prompt=(
            "You are a newsletter research editor. Use live search, deduplicate stories, "
            "verify important claims, and include direct source URLs."
        ),
        tools=[
            news_search(
                provider="haystack",
                default_params={"num": "7", "hl": "en", "gl": "us"},
            ),
            web_search(
                provider="haystack",
                allowed_engines=["google_light", "bing"],
                default_params={"num": "5", "hl": "en", "gl": "us"},
            ),
        ],
        max_agent_steps=8,
    )
    result = agent.run(messages=[ChatMessage.from_user(PROMPT)])
    report = _report_text(result)
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    main()
