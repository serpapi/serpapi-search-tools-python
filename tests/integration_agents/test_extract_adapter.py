import asyncio

import pytest
from optional_dependencies import import_optional

from serpapi_search_tools import web_crawl, web_fetch, web_map

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("provider", ["langchain", "langgraph"])
@pytest.mark.parametrize(
    "factory,operation", [(web_fetch, "scrape"), (web_map, "map"), (web_crawl, "crawl")]
)
def test_extract_native_sync_and_async_invocation(provider, factory, operation):
    import_optional("langchain_core.tools")
    calls = []

    class Client:
        def extract(self, params):
            calls.append(dict(params))
            return "# Page\n\n[Next](https://example.com/next)\n"

    tool = factory(provider=provider, client=Client())
    expected = "# Page\n\n[Next](https://example.com/next)\n"
    assert tool.invoke({"url": "https://example.com"}) == expected
    assert asyncio.run(tool.ainvoke({"url": "https://example.com"})) == expected
    assert [call["mode"] for call in calls] == [operation, operation]
    assert all(call["output"] == "md" and "json_restrictor" not in call for call in calls)
    assert tool.args_schema.model_json_schema()["required"] == ["url"]


def test_web_fetch_openai_agents_invocation():
    import_optional("agents")

    class Client:
        def extract(self, params):
            return "# Page\n\nContent.\n"

    tool = web_fetch(provider="openai-agents", client=Client())
    assert tool.params_json_schema["required"] == ["url"]
    assert asyncio.run(tool.on_invoke_tool(None, '{"url":"https://example.com"}')) == (
        "# Page\n\nContent.\n"
    )


def test_web_fetch_pydantic_ai_invocation_and_error_retry():
    import_optional("pydantic_ai")
    from pydantic_ai import ModelRetry

    class Client:
        def extract(self, params):
            if params["url"] == "https://example.com/error":
                return {"error": "Page fetch failed."}
            return "# Page\n\nContent.\n"

    tool = web_fetch(provider="pydantic-ai", client=Client())
    assert tool.function("https://example.com") == "# Page\n\nContent.\n"
    with pytest.raises(ModelRetry, match="SerpApi extraction failed"):
        tool.function("https://example.com/error")
