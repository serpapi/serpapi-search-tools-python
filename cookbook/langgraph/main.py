# Run: uv run --isolated --no-project --with 'serpapi-search-tools[langgraph]' --with python-dotenv --with langchain-openai cookbook/langgraph/main.py  # noqa: E501
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from serpapi_search_tools import news_search, shopping_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = os.getenv(
    "COOKBOOK_PROMPT",
    (
        "Investigate the current launch landscape for compact AI voice recorders. "
        "Use web search for official product facts, news search for recent launches and "
        "reviews, and shopping search for live price signals. Produce a concise launch "
        "intelligence brief with evidence, disagreements, and a recommendation."
    ),
)


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
    path = output_dir / "langgraph-launch-intelligence.md"
    path.write_text(text)
    return path


def main() -> None:
    _require_serpapi_key()
    tools = [
        web_search(
            provider="langgraph",
            allowed_engines=["google_light", "bing"],
            result_limit=5,
        ),
        news_search(
            provider="langgraph",
            default_params={"hl": "en", "gl": "us"},
            result_limit=5,
        ),
        shopping_search(
            provider="langgraph",
            result_limit=5,
        ),
    ]
    model = ChatOpenAI(
        model=MODEL,
        api_key=_require_env("OPENAI_API_KEY"),
        temperature=0,
    ).bind_tools(tools)

    def call_model(state: MessagesState) -> dict[str, list[Any]]:
        return {"messages": [model.invoke(state["messages"])]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("research", call_model)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_edge(START, "research")
    graph_builder.add_conditional_edges("research", tools_condition)
    graph_builder.add_edge("tools", "research")
    agent = graph_builder.compile()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Act as a launch-intelligence analyst. Use every relevant search "
                        "surface, reconcile conflicting evidence, and include source URLs."
                    ),
                },
                {"role": "user", "content": PROMPT},
            ]
        }
    )
    report = str(result["messages"][-1].content)
    path = _write_report(report)
    print(report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    main()
