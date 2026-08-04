from __future__ import annotations

import asyncio
import inspect
import json
import re
import sys
from collections.abc import Callable, Mapping
from enum import Enum
from importlib import import_module
from types import GenericAlias
from typing import Annotated, Any, Literal, cast, get_type_hints

from serpapi_search_tools._shared import (
    PROVIDER_ALIASES,
    ProviderName,
    SerpApiSearchError,
    ToolDefinition,
    detect_provider,
    normalize_provider,
)

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TOOL_INPUT_ERROR_LIMIT = 500
_PYTHON_VERSION = sys.version_info[:2]


def _dependency_error(extra: str, package: str, exc: ImportError) -> ImportError:
    message = (
        f"{package} is required for this adapter. Install it with "
        f"`pip install 'serpapi-search-tools[{extra}]'`."
    )
    error = ImportError(message)
    error.__cause__ = exc
    return error


def _direct_dependency_error(package: str, exc: ImportError) -> ImportError:
    error = ImportError(
        f"{package} is required for this adapter. Install it with `pip install {package}`."
    )
    error.__cause__ = exc
    return error


def _validate_tool_name(name: str) -> None:
    if not _TOOL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Tool names must contain 1 to 64 ASCII letters, digits, underscores, or hyphens."
        )


def _allows_none(
    name: str,
    schema: Mapping[str, Any],
    required: set[str],
) -> bool:
    return name not in required and schema.get("default") is None


def _with_tool_metadata(definition: ToolDefinition) -> Callable[..., str]:
    function = cast(Any, definition.function)
    function.__name__ = definition.name
    function.__qualname__ = definition.name
    function.__doc__ = definition.description
    function.__annotations__ = get_type_hints(function, include_extras=True)
    return cast(Callable[..., str], function)


def _pydantic_args_model(
    definition: ToolDefinition,
    *,
    extra: Literal["allow", "ignore", "forbid"] = "forbid",
) -> Any:
    try:
        from pydantic import ConfigDict, Field, create_model
    except ImportError as exc:
        raise _direct_dependency_error("pydantic", exc) from exc

    properties = cast(Mapping[str, Mapping[str, Any]], definition.input_schema["properties"])
    required = set(cast(list[str], definition.input_schema.get("required", [])))
    fields: dict[str, tuple[Any, Any]] = {}
    for name, schema in properties.items():
        base_type = _pydantic_type_from_schema(name, schema, Field)
        field_type = base_type
        if _allows_none(name, schema, required):
            field_type = field_type | None
        default = ... if name in required else schema.get("default")
        if default is not None and isinstance(base_type, type) and issubclass(base_type, Enum):
            default = base_type(default)
        constraints = _pydantic_field_constraints(schema)
        schema_extra = {"format": schema["format"]} if "format" in schema else None
        fields[name] = (
            field_type,
            Field(
                default,
                description=schema.get("description"),
                json_schema_extra=schema_extra,
                **constraints,
            ),
        )
    model_name = f"{''.join(part.title() for part in definition.name.split('_'))}Arguments"
    model_factory = cast(Callable[..., Any], create_model)
    return model_factory(
        model_name,
        __config__=ConfigDict(
            extra=extra,
            json_schema_extra={"additionalProperties": False},
        ),
        **fields,
    )


def _pydantic_field_constraints(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: schema[source]
        for key, source in (
            ("ge", "minimum"),
            ("le", "maximum"),
            ("min_length", "minLength"),
            ("max_length", "maxLength"),
        )
        if source in schema
    }


def _pydantic_type_from_schema(
    name: str,
    schema: Mapping[str, Any],
    field_factory: Callable[..., Any],
) -> Any:
    if schema.get("type") != "array":
        return _python_type_from_schema(name, schema)

    item_schema = cast(Mapping[str, Any], schema.get("items", {}))
    item_type = _python_type_from_schema(f"{name}_item", item_schema)
    item_constraints = _pydantic_field_constraints(item_schema)
    item_extra = {"format": item_schema["format"]} if "format" in item_schema else None
    if item_constraints or item_extra or item_schema.get("description"):
        item_type = Annotated[
            item_type,
            field_factory(
                description=item_schema.get("description"),
                json_schema_extra=item_extra,
                **item_constraints,
            ),
        ]
    return list[item_type]


def _with_pydantic_annotations(definition: ToolDefinition) -> Callable[..., str]:
    try:
        from pydantic import Field
    except ImportError as exc:
        raise _direct_dependency_error("pydantic", exc) from exc

    function = _with_tool_metadata(definition)
    properties = cast(Mapping[str, Mapping[str, Any]], definition.input_schema["properties"])
    required = set(cast(list[str], definition.input_schema.get("required", [])))
    annotations: dict[str, Any] = {}
    for name, schema in properties.items():
        field_type = _pydantic_type_from_schema(name, schema, Field)
        if _allows_none(name, schema, required):
            field_type = field_type | None
        schema_extra = {"format": schema["format"]} if "format" in schema else None
        annotations[name] = Annotated[
            field_type,
            str(schema.get("description") or name),
            Field(
                description=schema.get("description"),
                json_schema_extra=schema_extra,
                **_pydantic_field_constraints(schema),
            ),
        ]
    annotations["return"] = str
    function.__annotations__ = annotations
    return function


def _python_type_from_schema(name: str, schema: Mapping[str, Any]) -> Any:
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        enum_name = f"{''.join(part.title() for part in name.split('_'))}Value"
        return Enum(
            enum_name,
            {str(value).upper(): value for value in enum_values},
            type=str,
        )
    schema_type = schema.get("type")
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "object":
        return dict[str, Any]
    if schema_type == "array":
        items = cast(Mapping[str, Any], schema.get("items", {}))
        return GenericAlias(list, _python_type_from_schema(f"{name}_item", items))
    return Any


def as_function_tool(definition: ToolDefinition) -> Callable[..., str]:
    return _with_tool_metadata(definition)


def as_langchain_tool(definition: ToolDefinition) -> Any:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise _dependency_error("langchain", "langchain-core", exc) from exc

    return StructuredTool.from_function(
        func=definition.function,
        name=definition.name,
        description=definition.description,
        args_schema=_pydantic_args_model(definition),
    )


def as_crewai_tool(definition: ToolDefinition) -> Any:
    try:
        from crewai.tools import BaseTool
    except ImportError as exc:
        if _PYTHON_VERSION >= (3, 14):
            raise ImportError(
                "The CrewAI adapter supports Python 3.10 through 3.13. "
                "Use a supported Python version or choose another provider."
            ) from exc
        raise _dependency_error("crewai", "crewai", exc) from exc

    function = _with_tool_metadata(definition)

    def _run(self: Any, **kwargs: Any) -> str:
        return function(**kwargs)

    tool_type = type("SerpApiCrewTool", (BaseTool,), {"_run": _run})
    return tool_type(
        name=definition.name,
        description=definition.description,
        args_schema=_pydantic_args_model(definition, extra="ignore"),
    )


def as_llamaindex_tool(definition: ToolDefinition) -> Any:
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:
        raise _dependency_error("llamaindex", "llama-index-core", exc) from exc

    return FunctionTool.from_defaults(
        fn=definition.function,
        name=definition.name,
        description=definition.description,
        fn_schema=_pydantic_args_model(definition),
    )


def as_openai_agents_tool(definition: ToolDefinition) -> Any:
    try:
        from agents import FunctionTool
    except ImportError as exc:
        raise _dependency_error("openai-agents", "openai-agents", exc) from exc

    arguments_model = _pydantic_args_model(definition)

    async def invoke_tool(context: Any, arguments_json: str) -> str:
        try:
            validated = arguments_model.model_validate_json(arguments_json or "{}")
        except ValueError as exc:
            return _invalid_tool_arguments(exc)
        arguments = validated.model_dump(exclude_unset=True)
        try:
            return await asyncio.to_thread(definition.function, **arguments)
        except ValueError as exc:
            return _invalid_tool_arguments(exc)

    return FunctionTool(
        name=definition.name,
        description=definition.description,
        params_json_schema=definition.input_schema,
        on_invoke_tool=invoke_tool,
        strict_json_schema=False,
    )


def _invalid_tool_arguments(exc: ValueError) -> str:
    details: list[str] = []
    errors_method = getattr(exc, "errors", None)
    if callable(errors_method):
        try:
            errors = errors_method(include_url=False, include_input=False)
        except TypeError:
            errors = errors_method()
        for error in errors[:3]:
            location = ".".join(str(part) for part in error.get("loc", ()))
            message = str(error.get("msg", "invalid value"))
            details.append(f"{location}: {message}" if location else message)
    if not details:
        details.append(str(exc))
    detail = " ".join(" ".join(part.split()) for part in details)
    if len(detail) > _TOOL_INPUT_ERROR_LIMIT:
        detail = f"{detail[: _TOOL_INPUT_ERROR_LIMIT - 3]}..."
    return json.dumps(
        {"error": "Invalid tool arguments.", "details": detail},
        separators=(",", ":"),
    )


def as_claude_agent_sdk_tool(definition: ToolDefinition) -> Any:
    try:
        from claude_agent_sdk import ToolAnnotations, tool
    except ImportError as exc:
        raise _dependency_error("claude-agent-sdk", "claude-agent-sdk", exc) from exc

    async def run_tool(args: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        result = await asyncio.to_thread(definition.function, **args)
        return {"content": [{"type": "text", "text": result}]}

    return tool(
        definition.name,
        definition.description,
        definition.input_schema,
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    )(run_tool)


def as_autogen_tool(definition: ToolDefinition) -> Any:
    try:
        from autogen_core.tools import FunctionTool
    except ImportError as exc:
        raise _dependency_error("autogen", "autogen-core", exc) from exc

    native_tool = FunctionTool(
        _with_tool_metadata(definition),
        name=definition.name,
        description=definition.description,
    )
    native_tool._args_type = _pydantic_args_model(definition)
    return native_tool


def as_microsoft_agent_framework_tool(definition: ToolDefinition) -> Any:
    try:
        from agent_framework import FunctionTool
    except ImportError as exc:
        raise _dependency_error(
            "microsoft-agent-framework",
            "agent-framework-core",
            exc,
        ) from exc

    return FunctionTool(
        name=definition.name,
        description=definition.description,
        func=definition.function,
        input_model=definition.input_schema,
    )


def as_pydantic_ai_tool(definition: ToolDefinition) -> Any:
    try:
        pydantic_exceptions = cast(Any, import_module("pydantic_ai.exceptions"))
        pydantic_tools = cast(Any, import_module("pydantic_ai.tools"))
    except ImportError as exc:
        raise _dependency_error("pydantic-ai", "pydantic-ai", exc) from exc

    ModelRetry = pydantic_exceptions.ModelRetry
    Tool = pydantic_tools.Tool
    function = _with_pydantic_annotations(definition)

    def invoke(*args: Any, **kwargs: Any) -> str:
        try:
            return function(*args, **kwargs)
        except (ValueError, SerpApiSearchError) as exc:
            raise ModelRetry(str(exc)) from exc

    typed_invoke = cast(Any, invoke)
    typed_function = cast(Any, function)
    typed_invoke.__name__ = definition.name
    typed_invoke.__qualname__ = definition.name
    typed_invoke.__doc__ = definition.description
    typed_invoke.__annotations__ = typed_function.__annotations__
    typed_invoke.__signature__ = inspect.signature(function)

    return Tool(
        invoke,
        name=definition.name,
        description=definition.description,
    )


def as_haystack_tool(definition: ToolDefinition) -> Any:
    try:
        from haystack.tools import Tool
    except ImportError as exc:
        raise _dependency_error("haystack", "haystack-ai", exc) from exc

    return Tool(
        name=definition.name,
        description=definition.description,
        parameters=definition.input_schema,
        function=definition.function,
    )


def as_semantic_kernel_tool(definition: ToolDefinition) -> Any:
    try:
        from semantic_kernel.functions import kernel_function
    except ImportError as exc:
        raise _dependency_error("semantic-kernel", "semantic-kernel", exc) from exc

    function = _with_tool_metadata(definition)

    def invoke(**kwargs: Any) -> str:
        return function(**kwargs)

    decorated = kernel_function(name=definition.name, description=definition.description)(invoke)
    properties = cast(Mapping[str, Mapping[str, Any]], definition.input_schema["properties"])
    required = set(cast(list[str], definition.input_schema.get("required", [])))
    parameters: list[dict[str, Any]] = []
    for field_name, property_schema in properties.items():
        type_object = _python_type_from_schema(field_name, property_schema)
        parameter: dict[str, Any] = {
            "name": field_name,
            "description": property_schema.get("description", ""),
            "type_": getattr(type_object, "__name__", str(type_object)),
            "type_object": type_object,
            "is_required": field_name in required,
            "schema_data": dict(property_schema),
        }
        if field_name not in required:
            parameter["default_value"] = property_schema.get("default")
        parameters.append(parameter)

    decorated.__kernel_function_parameters__ = parameters
    decorated.__signature__ = inspect.signature(function)
    decorated.__annotations__ = function.__annotations__
    return decorated


def as_agno_tool(definition: ToolDefinition) -> Any:
    try:
        from agno.tools import Function
    except ImportError as exc:
        raise _dependency_error("agno", "agno", exc) from exc

    function = _with_tool_metadata(definition)
    return Function(
        name=definition.name,
        description=definition.description,
        parameters=definition.input_schema,
        entrypoint=function,
        strict=False,
    )


def _smolagents_inputs(schema: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    properties = cast(Mapping[str, Mapping[str, Any]], schema.get("properties", {}))
    required = set(cast(list[str], schema.get("required", [])))
    inputs: dict[str, dict[str, Any]] = {}
    for field_name, property_schema in properties.items():
        converted = dict(property_schema)
        converted.pop("additionalProperties", None)
        if field_name not in required:
            converted["nullable"] = True
        inputs[field_name] = converted
    return inputs


def as_smolagents_tool(definition: ToolDefinition) -> Any:
    try:
        from smolagents import Tool
    except ImportError as exc:
        raise _dependency_error("smolagents", "smolagents", exc) from exc

    def forward(self: Any, **kwargs: Any) -> str:
        return definition.function(**kwargs)

    original_signature = inspect.signature(definition.function)
    self_parameter = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    cast(Any, forward).__signature__ = original_signature.replace(
        parameters=[self_parameter, *original_signature.parameters.values()]
    )
    cast(Any, forward).__annotations__ = {
        "self": Any,
        **definition.function.__annotations__,
    }

    tool_type = type(
        "SerpApiSmolagentsTool",
        (Tool,),
        {
            "name": definition.name,
            "description": definition.description,
            "inputs": _smolagents_inputs(definition.input_schema),
            "output_type": "string",
            "forward": forward,
        },
    )
    return tool_type()


def as_google_adk_tool(definition: ToolDefinition) -> Any:
    try:
        from google.adk.tools import FunctionTool
    except ImportError as exc:
        raise _dependency_error("google-adk", "google-adk", exc) from exc

    class SerpApiFunctionTool(FunctionTool):
        def _get_declaration(self: Any) -> Any:
            declaration = super()._get_declaration()
            if declaration is not None:
                declaration.parameters_json_schema = definition.input_schema
            return declaration

    return SerpApiFunctionTool(_with_tool_metadata(definition))


def as_provider_tool(
    definition: ToolDefinition,
    provider: ProviderName | str = "auto",
) -> Any:
    """Adapt a provider-neutral definition into one SDK-native tool."""

    _validate_tool_name(definition.name)
    definition = ToolDefinition(
        function=_with_tool_metadata(definition),
        name=definition.name,
        description=definition.description,
        input_schema=definition.input_schema,
    )
    normalized = normalize_provider(provider)
    if normalized == "auto":
        normalized = detect_provider()

    adapters: dict[str, Callable[[ToolDefinition], Any]] = {
        "function": as_function_tool,
        "langchain": as_langchain_tool,
        "langgraph": as_langchain_tool,
        "crewai": as_crewai_tool,
        "llamaindex": as_llamaindex_tool,
        "openai-agents": as_openai_agents_tool,
        "claude-agent-sdk": as_claude_agent_sdk_tool,
        "pydantic-ai": as_pydantic_ai_tool,
        "microsoft-agent-framework": as_microsoft_agent_framework_tool,
        "autogen": as_autogen_tool,
        "haystack": as_haystack_tool,
        "semantic-kernel": as_semantic_kernel_tool,
        "agno": as_agno_tool,
        "smolagents": as_smolagents_tool,
        "google-adk": as_google_adk_tool,
    }
    adapter = adapters.get(normalized)
    if adapter is None:
        supported = ", ".join(PROVIDER_ALIASES)
        raise ValueError(f"Unknown provider '{provider}'. Supported providers: {supported}.")
    return adapter(definition)
