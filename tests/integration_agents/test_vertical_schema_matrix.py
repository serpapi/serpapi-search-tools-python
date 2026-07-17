from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from optional_dependencies import import_optional

from serpapi_search_tools import (
    flights_search,
    hotels_search,
    images_search,
    maps_search,
    news_search,
    shopping_search,
    travel_explore_search,
    videos_search,
    web_search,
)

pytestmark = pytest.mark.integration

PROVIDER_MODULES = {
    "agno": "agno.tools",
    "autogen": "autogen_core.tools",
    "claude-agent-sdk": "claude_agent_sdk",
    "crewai": "crewai.tools",
    "google-adk": "google.adk.tools",
    "haystack": "haystack.tools",
    "langchain": "langchain_core.tools",
    "langgraph": "langchain_core.tools",
    "llamaindex": "llama_index.core.tools",
    "openai-agents": "agents",
    "pydantic-ai": "pydantic_ai.tools",
    "semantic-kernel": "semantic_kernel.functions",
    "smolagents": "smolagents",
}


class NoSearchClient:
    def search(self, params: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"schema construction must not call SerpApi: {params}")


SCHEMA_CASES: list[tuple[Callable[..., Any], set[str], set[str]]] = [
    (web_search, {"query", "engine"}, {"query"}),
    (news_search, {"query"}, {"query"}),
    (maps_search, {"query", "location", "zoom", "nearby"}, {"query"}),
    (images_search, {"query"}, {"query"}),
    (shopping_search, {"query", "engine"}, {"query"}),
    (videos_search, {"query"}, {"query"}),
    (
        hotels_search,
        {"query", "check_in_date", "check_out_date", "adults", "children", "children_ages"},
        {"query", "check_in_date", "check_out_date"},
    ),
    (
        flights_search,
        {
            "departure_id",
            "arrival_id",
            "outbound_date",
            "return_date",
            "travel_class",
            "adults",
            "children",
            "infants_in_seat",
            "infants_on_lap",
        },
        {"departure_id", "arrival_id", "outbound_date"},
    ),
    (
        travel_explore_search,
        {
            "departure_id",
            "arrival_id",
            "arrival_area_id",
            "outbound_date",
            "return_date",
            "travel_class",
            "adults",
            "children",
            "infants_in_seat",
            "infants_on_lap",
        },
        {"departure_id"},
    ),
]


@pytest.mark.parametrize(("provider", "module_name"), PROVIDER_MODULES.items())
@pytest.mark.parametrize(("factory", "properties", "required"), SCHEMA_CASES)
def test_installed_sdk_preserves_each_public_input_shape(
    provider: str,
    module_name: str,
    factory: Callable[..., Any],
    properties: set[str],
    required: set[str],
) -> None:
    import_optional(module_name)
    kwargs: dict[str, Any] = {
        "provider": provider,
        "client": NoSearchClient(),
        "include_examples": False,
    }
    if factory is web_search:
        kwargs.update(allowed_engines=["google_light", "bing"], default_engine="bing")
    tool = factory(**kwargs)
    schema = _normalized_schema(provider, tool)

    assert set(schema["properties"]) == properties
    assert set(schema["required"]) == required
    assert "search_query" not in schema["properties"]
    assert "serpapi_params" not in schema["properties"]
    if factory in {flights_search, travel_explore_search}:
        assert "query" not in schema["properties"]


@pytest.mark.parametrize(("provider", "module_name"), PROVIDER_MODULES.items())
def test_installed_sdk_preserves_enums_defaults_arrays_and_date_fields(
    provider: str,
    module_name: str,
) -> None:
    import_optional(module_name)

    web_schema = _normalized_schema(
        provider,
        web_search(
            provider=provider,
            allowed_engines=["google_light", "bing"],
            default_engine="bing",
            client=NoSearchClient(),
        ),
    )
    flight_schema = _normalized_schema(
        provider,
        flights_search(provider=provider, client=NoSearchClient()),
    )
    hotel_schema = _normalized_schema(
        provider,
        hotels_search(provider=provider, client=NoSearchClient()),
    )

    assert _enum_values(web_schema, "engine") == {"google_light", "bing"}
    assert _enum_values(flight_schema, "travel_class") == {
        "economy",
        "premium_economy",
        "business",
        "first",
    }
    assert _property_type(hotel_schema, "children_ages") == "array"
    assert _property_type(hotel_schema, "check_in_date") == "string"
    assert _property_schema(hotel_schema, "check_in_date")["format"] == "date"
    age_items = _resolve_schema(
        hotel_schema,
        _property_schema(hotel_schema, "children_ages")["items"],
    )
    assert age_items["minimum"] == 1
    assert age_items["maximum"] == 17

    maps_schema = _normalized_schema(
        provider,
        maps_search(provider=provider, client=NoSearchClient()),
    )
    zoom_schema = _property_schema(maps_schema, "zoom")
    assert zoom_schema["minimum"] == 3
    assert zoom_schema["maximum"] == 30

    for schema in (web_schema, flight_schema, hotel_schema, maps_schema):
        assert schema.get("additionalProperties") is False


def _normalized_schema(provider: str, tool: Any) -> dict[str, Any]:
    if provider in {"langchain", "langgraph", "crewai"}:
        return tool.args_schema.model_json_schema()
    if provider == "llamaindex":
        return tool.metadata.fn_schema.model_json_schema()
    if provider == "openai-agents":
        return tool.params_json_schema
    if provider == "claude-agent-sdk":
        return tool.input_schema
    if provider == "pydantic-ai":
        if hasattr(tool, "function_schema"):
            return tool.function_schema.json_schema
        pydantic_tools = import_optional("pydantic_ai.tools")
        return pydantic_tools.Tool(tool).function_schema.json_schema
    if provider == "autogen":
        return tool.schema["parameters"]
    if provider == "haystack":
        return tool.parameters
    if provider == "semantic-kernel":
        return _semantic_kernel_schema(tool)
    if provider == "agno":
        return tool.parameters
    if provider == "smolagents":
        required = [
            name
            for name, schema in tool.inputs.items()
            if not schema.get("nullable") and "default" not in schema
        ]
        return {
            "type": "object",
            "properties": tool.inputs,
            "required": required,
            "additionalProperties": False,
        }
    if provider == "google-adk":
        return tool._get_declaration().parameters_json_schema
    raise AssertionError(f"Missing schema normalizer for {provider}")


def _semantic_kernel_schema(tool: Any) -> dict[str, Any]:
    functions = import_optional("semantic_kernel.functions")
    native_function = functions.KernelFunctionFromMethod(tool, plugin_name="search")
    properties: dict[str, Any] = {}
    required: list[str] = []
    for metadata in native_function.metadata.parameters:
        parameter = metadata.model_dump()
        name = parameter["name"]
        if parameter.get("schema_data"):
            schema = dict(parameter["schema_data"])
            properties[name] = schema
            if parameter.get("is_required"):
                required.append(name)
            continue
        type_object = parameter.get("type_object")
        schema: dict[str, Any]
        if isinstance(type_object, type) and issubclass(type_object, __import__("enum").Enum):
            schema = {"type": "string", "enum": [member.value for member in type_object]}
        elif type_object is str:
            schema = {"type": "string"}
        elif type_object is int:
            schema = {"type": "integer"}
        elif type_object is bool:
            schema = {"type": "boolean"}
        elif getattr(type_object, "__origin__", None) is list:
            schema = {"type": "array", "items": {"type": "integer"}}
        else:
            schema = {"type": "string"}
        if "default_value" in parameter:
            default = parameter["default_value"]
            schema["default"] = getattr(default, "value", default)
        properties[name] = schema
        if parameter.get("is_required"):
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _resolve_schema(root: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if ref:
        resolved: Any = root
        for part in ref.removeprefix("#/").split("/"):
            resolved = resolved[part]
        return resolved
    for candidate in schema.get("anyOf", []):
        if candidate.get("type") != "null":
            return _resolve_schema(root, candidate)
    return schema


def _enum_values(schema: dict[str, Any], name: str) -> set[str]:
    property_schema = _resolve_schema(schema, schema["properties"][name])
    return set(property_schema["enum"])


def _property_type(schema: dict[str, Any], name: str) -> str:
    return str(_property_schema(schema, name)["type"])


def _property_schema(schema: dict[str, Any], name: str) -> dict[str, Any]:
    return _resolve_schema(schema, schema["properties"][name])
