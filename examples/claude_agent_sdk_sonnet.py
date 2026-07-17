# Run: uv run --with . --with claude-agent-sdk examples/claude_agent_sdk_sonnet.py
from __future__ import annotations

import asyncio
import os

from _logging_client import LoggingClient
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query
from dotenv import load_dotenv

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
PROMPT = (
    "Use the SerpApi MCP tool to search Google News for 'portable solar generator'. "
    "Return three concise findings."
)


def _require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"Set {name} before running this example.")


def _require_serpapi_key() -> str:
    value = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not value:
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")
    return value


async def main() -> None:
    _require_env("ANTHROPIC_API_KEY")
    client = LoggingClient(_require_serpapi_key(), max_results=3)
    serpapi_server = create_sdk_mcp_server(
        name="serpapi",
        version="1.0.0",
        tools=[
            news_search(
                provider="claude-agent-sdk",
                default_params={"num": "3", "hl": "en", "gl": "us"},
                client=client,
            ),
            web_search(
                provider="claude-agent-sdk",
                allowed_engines=["google_light"],
                client=client,
            ),
        ],
    )
    options = ClaudeAgentOptions(
        model=MODEL,
        mcp_servers={"serpapi": serpapi_server},
        allowed_tools=["mcp__serpapi__news_search", "mcp__serpapi__web_search"],
        max_turns=6,
    )
    async for message in query(prompt=PROMPT, options=options):
        result = getattr(message, "result", None)
        if result:
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
