# Run: uv run --isolated --no-project --with 'serpapi-search-tools[langgraph]' --with python-dotenv --with langchain-openai examples/langgraph_openai.py  # noqa: E501
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from serpapi_search_tools import news_search, web_search

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PROMPT = (
    "Use the SerpApi tool to search Google for 'latest Python packaging best practices'. "
    "Summarize the useful guidance."
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
    search_tools = [
        web_search(
            provider="langgraph",
            allowed_engines=["google_light", "google"],
            default_params={"hl": "en", "gl": "us"},
            result_limit=3,
        ),
        news_search(
            provider="langgraph",
            default_params={"hl": "en", "gl": "us"},
            result_limit=3,
        ),
    ]
    model = ChatOpenAI(
        model=MODEL,
        api_key=_require_env("OPENAI_API_KEY"),
        temperature=0,
    ).bind_tools(search_tools)

    def call_model(state: MessagesState) -> dict[str, list[Any]]:
        return {"messages": [model.invoke(state["messages"])]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("model", call_model)
    graph_builder.add_node("tools", ToolNode(search_tools))
    graph_builder.add_edge(START, "model")
    graph_builder.add_conditional_edges("model", tools_condition)
    graph_builder.add_edge("tools", "model")
    agent = graph_builder.compile()

    result = agent.invoke({"messages": [{"role": "user", "content": PROMPT}]})
    final_message = result["messages"][-1]
    print(getattr(final_message, "content", final_message))


if __name__ == "__main__":
    main()
