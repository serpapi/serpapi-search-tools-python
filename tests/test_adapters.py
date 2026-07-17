import asyncio
import builtins
import inspect
import json
import sys
import threading
from types import ModuleType
from typing import Any

import pytest

from serpapi_search_tools import flights_search, hotels_search, maps_search, web_search


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(self, params: dict[str, object]) -> dict[str, object]:
        self.calls.append(params)
        return {"params": params}


class CoordinatedClient:
    def __init__(
        self,
        events: list[str],
        search_started: threading.Event,
        release_search: threading.Event,
    ) -> None:
        self.events = events
        self.search_started = search_started
        self.release_search = release_search

    def search(self, params: dict[str, object]) -> dict[str, object]:
        self.search_started.set()
        self.release_search.wait(timeout=1)
        self.events.append("search-finished")
        return {"params": params}


class StructuredTool:
    @classmethod
    def from_function(cls, *, func, name: str, description: str, args_schema):
        return {
            "kind": "langchain",
            "func": func,
            "name": name,
            "description": description,
            "args_schema": args_schema,
        }


class LlamaIndexFunctionTool:
    @classmethod
    def from_defaults(cls, *, fn, name: str, description: str, fn_schema):
        return {
            "kind": "llamaindex",
            "func": fn,
            "name": name,
            "description": description,
            "fn_schema": fn_schema,
        }


def test_langchain_adapter_preserves_structured_hotel_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("langchain_core.tools")
    module.StructuredTool = StructuredTool
    monkeypatch.setitem(sys.modules, "langchain_core.tools", module)
    client = FakeClient()

    tool = hotels_search(provider="langchain", client=client)
    result = json.loads(
        tool["func"](
            query="Kyoto hotels",
            check_in_date="2026-08-01",
            check_out_date="2026-08-04",
        )
    )

    assert tool["name"] == "hotels_search"
    assert list(inspect.signature(tool["func"]).parameters)[:3] == [
        "query",
        "check_in_date",
        "check_out_date",
    ]
    assert result["params"]["engine"] == "google_hotels"


def test_langgraph_adapter_reuses_langchain_native_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("langchain_core.tools")
    module.StructuredTool = StructuredTool
    monkeypatch.setitem(sys.modules, "langchain_core.tools", module)

    tool = maps_search(provider="langgraph", client=FakeClient())

    assert tool["kind"] == "langchain"
    assert tool["name"] == "maps_search"


def test_llamaindex_adapter_preserves_flight_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("llama_index.core.tools")
    module.FunctionTool = LlamaIndexFunctionTool
    monkeypatch.setitem(sys.modules, "llama_index.core.tools", module)

    tool = flights_search(provider="llamaindex", client=FakeClient())

    assert tool["kind"] == "llamaindex"
    assert list(inspect.signature(tool["func"]).parameters)[:3] == [
        "departure_id",
        "arrival_id",
        "outbound_date",
    ]


def test_crewai_adapter_uses_tool_decorator(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("crewai.tools")

    class BaseTool:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.BaseTool = BaseTool
    monkeypatch.setitem(sys.modules, "crewai.tools", module)

    tool_value = maps_search(provider="crewai", client=FakeClient())

    assert isinstance(tool_value, BaseTool)
    assert tool_value.name == "maps_search"
    assert "location" in tool_value.args_schema.model_fields
    validated = tool_value.args_schema.model_validate(
        {"query": "coffee", "security_context": {"agent": "test"}}
    )
    assert validated.model_dump(exclude_none=True) == {
        "query": "coffee",
        "zoom": 14,
        "nearby": False,
    }


def test_openai_agents_adapter_preserves_flight_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("agents")

    class FunctionTool:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.FunctionTool = FunctionTool
    monkeypatch.setitem(sys.modules, "agents", module)

    tool = flights_search(provider="openai-agents", client=FakeClient())

    assert tool.name == "flights_search"
    assert tool.strict_json_schema is False
    assert tool.params_json_schema["required"] == [
        "departure_id",
        "arrival_id",
        "outbound_date",
    ]
    assert "query" not in tool.params_json_schema["properties"]


def test_openai_agents_adapter_treats_empty_arguments_as_an_empty_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("agents")

    class FunctionTool:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.FunctionTool = FunctionTool
    monkeypatch.setitem(sys.modules, "agents", module)
    tool = web_search(provider="openai-agents", client=FakeClient())

    with pytest.raises(TypeError, match="query"):
        asyncio.run(tool.on_invoke_tool(None, ""))


def test_openai_agents_adapter_does_not_block_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("agents")

    class FunctionTool:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.FunctionTool = FunctionTool
    monkeypatch.setitem(sys.modules, "agents", module)
    events: list[str] = []
    search_started = threading.Event()
    release_search = threading.Event()
    tool = web_search(
        provider="openai-agents",
        client=CoordinatedClient(events, search_started, release_search),
    )

    async def tick() -> None:
        while not search_started.is_set():
            await asyncio.sleep(0)
        events.append("event-loop-tick")
        release_search.set()

    async def run() -> None:
        await asyncio.gather(
            tool.on_invoke_tool(None, json.dumps({"query": "coffee"})),
            tick(),
        )

    asyncio.run(run())

    assert events == ["event-loop-tick", "search-finished"]


def test_claude_adapter_uses_explicit_hotel_schema_and_forwards_arbitrary_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("claude_agent_sdk")

    class ToolAnnotations:
        def __init__(self, **kwargs: object) -> None:
            self.values = kwargs

    def tool(name: str, description: str, schema: dict[str, Any], *, annotations):
        def decorate(func):
            return {
                "name": name,
                "description": description,
                "schema": schema,
                "func": func,
                "annotations": annotations,
            }

        return decorate

    module.ToolAnnotations = ToolAnnotations
    module.tool = tool
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    client = FakeClient()

    tool_value = hotels_search(provider="claude-agent-sdk", client=client)
    result = asyncio.run(
        tool_value["func"](
            {
                "query": "Kyoto hotels",
                "check_in_date": "2026-08-01",
                "check_out_date": "2026-08-04",
                "children": 1,
                "children_ages": [8],
            }
        )
    )

    assert tool_value["schema"]["required"] == [
        "query",
        "check_in_date",
        "check_out_date",
    ]
    assert tool_value["schema"]["properties"]["children_ages"]["type"] == "array"
    assert json.loads(result["content"][0]["text"])["params"]["children_ages"] == "8"


def test_claude_adapter_does_not_block_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("claude_agent_sdk")

    class ToolAnnotations:
        def __init__(self, **kwargs: object) -> None:
            self.values = kwargs

    def tool(name: str, description: str, schema: dict[str, Any], *, annotations):
        def decorate(func):
            return {"func": func}

        return decorate

    module.ToolAnnotations = ToolAnnotations
    module.tool = tool
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    events: list[str] = []
    search_started = threading.Event()
    release_search = threading.Event()
    tool_value = web_search(
        provider="claude-agent-sdk",
        client=CoordinatedClient(events, search_started, release_search),
    )

    async def tick() -> None:
        while not search_started.is_set():
            await asyncio.sleep(0)
        events.append("event-loop-tick")
        release_search.set()

    async def run() -> None:
        await asyncio.gather(
            tool_value["func"]({"query": "coffee"}),
            tick(),
        )

    asyncio.run(run())

    assert events == ["event-loop-tick", "search-finished"]


def test_autogen_adapter_accepts_maps_function(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("autogen_core.tools")

    class FunctionTool:
        def __init__(self, func, *, name: str, description: str) -> None:
            self.func = func
            self.name = name
            self.description = description

    module.FunctionTool = FunctionTool
    monkeypatch.setitem(sys.modules, "autogen_core.tools", module)

    tool = maps_search(provider="autogen", client=FakeClient())

    assert tool.name == "maps_search"
    assert "location" in inspect.signature(tool.func).parameters


def test_haystack_adapter_accepts_web_function(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("haystack.tools")

    class Tool:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.Tool = Tool
    monkeypatch.setitem(sys.modules, "haystack.tools", module)

    tool = web_search(provider="haystack", client=FakeClient())

    assert tool.name == "web_search"
    assert tool.parameters["properties"]["engine"]["default"] == "google_light"


def test_semantic_kernel_decorates_original_structured_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("semantic_kernel.functions")

    def kernel_function(*, name: str, description: str):
        def decorate(func):
            func.semantic_kernel_name = name
            func.semantic_kernel_description = description
            return func

        return decorate

    module.kernel_function = kernel_function
    monkeypatch.setitem(sys.modules, "semantic_kernel.functions", module)

    tool = flights_search(provider="semantic-kernel", client=FakeClient())

    assert tool.semantic_kernel_name == "flights_search"
    assert list(inspect.signature(tool).parameters)[:3] == [
        "departure_id",
        "arrival_id",
        "outbound_date",
    ]


def test_agno_adapter_accepts_hotels_function(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("agno.tools")

    class Function:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.Function = Function
    monkeypatch.setitem(sys.modules, "agno.tools", module)

    tool = hotels_search(provider="agno", client=FakeClient())

    assert tool.name == "hotels_search"
    assert tool.strict is False
    assert tool.parameters["required"] == ["query", "check_in_date", "check_out_date"]


def test_smolagents_converts_flight_schema_and_forwards_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("smolagents")

    class Tool:
        def __init__(self) -> None:
            self.is_initialized = False

    module.Tool = Tool
    monkeypatch.setitem(sys.modules, "smolagents", module)
    client = FakeClient()

    tool = flights_search(provider="smolagents", client=client)
    encoded = tool.forward(
        departure_id="LAX",
        arrival_id="AUS",
        outbound_date="2026-08-01",
    )

    assert tool.name == "flights_search"
    assert tool.inputs["travel_class"]["enum"] == [
        "economy",
        "premium_economy",
        "business",
        "first",
    ]
    assert tool.inputs["return_date"]["nullable"] is True
    assert json.loads(encoded)["params"]["type"] == 2


def test_google_adk_adapter_preserves_maps_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("google.adk.tools")

    class FunctionTool:
        def __init__(self, func) -> None:
            self.func = func

    module.FunctionTool = FunctionTool
    monkeypatch.setitem(sys.modules, "google.adk.tools", module)

    tool = maps_search(provider="google-adk", client=FakeClient())

    assert tool.func.__name__ == "maps_search"
    assert "location" in inspect.signature(tool.func).parameters


def test_pydantic_ai_adapter_returns_native_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("pydantic_ai.tools")

    class Tool:
        def __init__(self, function, **kwargs: object) -> None:
            self.function = function
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.Tool = Tool
    monkeypatch.setitem(sys.modules, "pydantic_ai.tools", module)

    tool = hotels_search(provider="pydantic-ai", client=FakeClient())

    assert isinstance(tool, Tool)
    assert tool.name == "hotels_search"
    assert "check_in_date" in inspect.signature(tool.function).parameters


def test_missing_optional_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "langchain_core.tools":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="serpapi-search-tools\\[langchain\\]"):
        web_search(provider="langchain", client=FakeClient())


def test_unknown_provider_lists_supported_options() -> None:
    with pytest.raises(ValueError, match="Unknown provider 'future-sdk'"):
        web_search(provider="future-sdk", client=FakeClient())
