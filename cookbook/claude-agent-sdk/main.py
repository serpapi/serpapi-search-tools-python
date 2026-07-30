# Run: uv run --with . --with claude-agent-sdk cookbook/claude-agent-sdk/main.py
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query
from dotenv import load_dotenv

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Verify the claim that right-to-repair rules are rapidly converging across the "
        "United States. Use SerpApi web search for primary or durable sources and news "
        "search for recent developments. Produce a verification memo that separates "
        "confirmed facts, overstatements, jurisdiction differences, and source URLs."
    ),
)


def _require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"Set {name} before running this cookbook.")


def _require_serpapi_key() -> str:
    value = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not value:
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this cookbook.")
    return value


def _write_report(text: str) -> Path:
    output_dir = Path(os.getenv("COOKBOOK_OUTPUT_DIR", "cookbook-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "claude-source-verification.md"
    path.write_text(text)
    return path


async def main() -> None:
    _require_serpapi_key()
    _require_env("ANTHROPIC_API_KEY")
    serpapi_server = create_sdk_mcp_server(
        name="serpapi",
        version="1.0.0",
        tools=[
            web_search(
                provider="claude-agent-sdk",
                allowed_engines=["google_light", "bing"],
                default_params={"num": "5", "hl": "en", "gl": "us"},
            ),
            news_search(
                provider="claude-agent-sdk",
                default_params={"num": "5", "hl": "en", "gl": "us"},
            ),
        ],
    )
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=(
            "You are a source-verification analyst. Search before concluding, prefer "
            "primary sources when available, and expose conflicting evidence."
        ),
        mcp_servers={"serpapi": serpapi_server},
        allowed_tools=["mcp__serpapi__web_search", "mcp__serpapi__news_search"],
        max_turns=16,
    )
    reports: list[str] = []
    async for message in query(prompt=PROMPT, options=options):
        result = getattr(message, "result", None)
        if result:
            reports.append(str(result))
    if not reports:
        raise RuntimeError("Claude Agent SDK completed without a final report.")
    report = reports[-1]
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    asyncio.run(main())
