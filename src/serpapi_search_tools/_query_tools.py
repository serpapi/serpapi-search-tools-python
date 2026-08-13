from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, cast

from serpapi_search_tools._adapters import as_provider_tool
from serpapi_search_tools._shared import (
    ProviderName,
    SearchClient,
    SearchResultMode,
    SearchRuntime,
    ToolDefinition,
    object_schema,
    require_nonempty,
)


class WebSearchEngine(str, Enum):
    """General web indexes supported by :func:`web_search`."""

    GOOGLE = "google"
    GOOGLE_LIGHT = "google_light"
    BING = "bing"
    YAHOO = "yahoo"
    DUCKDUCKGO = "duckduckgo"


class ShoppingSearchEngine(str, Enum):
    """Product indexes supported by :func:`shopping_search`."""

    GOOGLE_SHOPPING = "google_shopping"
    AMAZON = "amazon"
    WALMART = "walmart"
    EBAY = "ebay"


WEB_QUERY_PARAM_BY_ENGINE: Mapping[str, str] = {
    "google": "q",
    "google_light": "q",
    "bing": "q",
    "yahoo": "p",
    "duckduckgo": "q",
}

SHOPPING_QUERY_PARAM_BY_ENGINE: Mapping[str, str] = {
    "google_shopping": "q",
    "amazon": "k",
    "walmart": "query",
    "ebay": "_nkw",
}

_VERTICAL_HINTS: Mapping[str, str] = {
    "google_news": "news_search",
    "google_maps": "maps_search",
    "google_images": "images_search",
    "youtube": "videos_search",
    "google_shopping": "shopping_search",
    "amazon": "shopping_search",
    "walmart": "shopping_search",
    "ebay": "shopping_search",
    "google_hotels": "hotels_search",
    "google_flights": "flights_search",
    "google_travel_explore": "travel_explore_search",
    "google": "web_search",
    "google_light": "web_search",
    "bing": "web_search",
    "yahoo": "web_search",
    "duckduckgo": "web_search",
}


def _runtime(
    *,
    api_key: str | None,
    client: SearchClient | None,
    default_params: Mapping[str, Any] | None,
    timeout: float | None,
    mode: SearchResultMode | str,
    result_limit: int | None,
) -> SearchRuntime:
    return SearchRuntime(
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
        result_limit=result_limit,
    )


def _normalize_allowed_engines(
    allowed_engines: Iterable[str | Enum] | str | Enum | None,
    *,
    catalog: type[Enum],
    tool_name: str,
) -> tuple[str, ...]:
    supported = tuple(cast(str, member.value) for member in catalog)
    if allowed_engines is None:
        return supported
    if isinstance(allowed_engines, (str, Enum)):
        raw_engines: tuple[str | Enum, ...] = (allowed_engines,)
    else:
        raw_engines = tuple(allowed_engines)
    if not raw_engines:
        raise ValueError(f"{tool_name} requires at least one allowed engine.")

    normalized = tuple(
        dict.fromkeys(
            cast(str, engine.value) if isinstance(engine, Enum) else engine
            for engine in raw_engines
        )
    )
    for engine in normalized:
        if engine not in supported:
            hint = _VERTICAL_HINTS.get(engine)
            if hint and hint != tool_name:
                raise ValueError(
                    f"Engine '{engine}' is not supported by {tool_name}. Use {hint}() instead."
                )
            choices = ", ".join(supported)
            raise ValueError(
                f"Engine '{engine}' is not supported by {tool_name}. Supported: {choices}."
            )
    return normalized


def _normalize_default_engine(
    default_engine: str | Enum | None,
    *,
    allowed_engines: tuple[str, ...],
    preferred: str,
    tool_name: str,
) -> str:
    if default_engine is None:
        return preferred if preferred in allowed_engines else allowed_engines[0]
    normalized = (
        cast(str, default_engine.value) if isinstance(default_engine, Enum) else default_engine
    )
    if normalized not in allowed_engines:
        choices = ", ".join(allowed_engines)
        raise ValueError(
            f"Default engine '{normalized}' is not allowed by {tool_name}. Allowed: {choices}."
        )
    return normalized


def _multi_engine_query_definition(
    *,
    name: str,
    description: str,
    catalog: type[Enum],
    query_params: Mapping[str, str],
    allowed_engines: Iterable[str | Enum] | str | Enum | None,
    default_engine: str | Enum | None,
    preferred_engine: str,
    engine_purposes: Mapping[str, str],
    runtime: SearchRuntime,
    include_examples: bool,
) -> ToolDefinition:
    allowed = _normalize_allowed_engines(
        allowed_engines,
        catalog=catalog,
        tool_name=name,
    )
    selected_default = _normalize_default_engine(
        default_engine,
        allowed_engines=allowed,
        preferred=preferred_engine,
        tool_name=name,
    )
    AllowedEngine = Enum(
        "AllowedEngine",
        {engine.upper(): engine for engine in allowed},
        type=str,
    )
    allowed_engine_type = AllowedEngine
    default_member = AllowedEngine(selected_default)

    def query_tool(
        query: str,
        engine: Any = default_member,
    ) -> str:
        normalized_engine = cast(str, engine.value) if isinstance(engine, Enum) else engine
        if normalized_engine not in allowed:
            choices = ", ".join(allowed)
            raise ValueError(
                f"Engine '{normalized_engine}' is not allowed by {name}. Allowed: {choices}."
            )
        native_query = query_params[normalized_engine]

        def validate(params: dict[str, Any]) -> None:
            require_nonempty(params, native_query, label="query")
            _validate_engine_mode(normalized_engine, params)

        return runtime.execute(
            engine=normalized_engine,
            typed_params={native_query: query},
            validator=validate,
        )

    engines_text = ", ".join(allowed)
    complete_description = f"{description} Supported engines: {engines_text}."
    if include_examples:
        complete_description += f" Example: query='coffee', engine='{selected_default}'."
    query_tool.__annotations__ = {
        "query": str,
        "engine": allowed_engine_type,
        "return": str,
    }
    schema = object_schema(
        {
            "query": {"type": "string", "description": "The search query."},
            "engine": {
                "type": "string",
                "enum": list(allowed),
                "default": selected_default,
                "description": (
                    f"Engine; defaults to {selected_default}. "
                    + "; ".join(f"{engine}: {engine_purposes[engine]}" for engine in allowed)
                    + "."
                ),
            },
        },
        required=["query"],
    )
    return ToolDefinition(query_tool, name, complete_description, schema)


def _fixed_query_definition(
    *,
    name: str,
    engine: str,
    query_param: str,
    description: str,
    runtime: SearchRuntime,
    include_examples: bool,
    validator: Any = None,
) -> ToolDefinition:
    def query_tool(query: str) -> str:
        def validate(params: dict[str, Any]) -> None:
            require_nonempty(params, query_param, label="query")
            if validator is not None:
                validator(params)

        return runtime.execute(
            engine=engine,
            typed_params={query_param: query},
            validator=validate,
        )

    complete_description = description
    if include_examples:
        complete_description += " Example: query='coffee'."
    query_tool.__annotations__ = {"query": str, "return": str}
    schema = object_schema(
        {
            "query": {"type": "string", "description": "The search query."},
        },
        required=["query"],
    )
    return ToolDefinition(query_tool, name, complete_description, schema)


def web_search(
    *,
    provider: ProviderName | str = "auto",
    allowed_engines: Iterable[WebSearchEngine | str] | WebSearchEngine | str | None = None,
    default_engine: WebSearchEngine | str | None = None,
    include_examples: bool = True,
    api_key: str | None = None,
    client: SearchClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    result_limit: int | None = 10,
    name: str = "web_search",
) -> Any:
    """Create a general-web search tool.

    Parameters
    ----------
    provider
        Agent SDK adapter name, ``"auto"``, or ``"function"`` for a callable.
    allowed_engines
        Web indexes exposed in the model-facing ``engine`` enum.
    default_engine
        Engine used when the model omits ``engine``. Defaults to Google Light
        when it is allowed.
    include_examples
        Add a short call example to the model-facing description.
    api_key, client, default_params, timeout, mode
        Runtime authentication, custom client, application filters, timeout, and
        compact or full response selection.
    result_limit
        Maximum items kept in each result list in both response modes; use
        ``None`` to keep all returned results.
    name
        Tool name presented to the model.

    Returns
    -------
    Any
        A provider-native tool whose input fields are ``query`` and ``engine``.
    """

    definition = _multi_engine_query_definition(
        name=name,
        description="Search general web pages and return focused SerpApi results as JSON.",
        catalog=WebSearchEngine,
        query_params=WEB_QUERY_PARAM_BY_ENGINE,
        allowed_engines=allowed_engines,
        default_engine=default_engine,
        preferred_engine=WebSearchEngine.GOOGLE_LIGHT.value,
        engine_purposes={
            "google": "rich results",
            "google_light": "fast results",
            "bing": "web index",
            "yahoo": "web index",
            "duckduckgo": "web index",
        },
        runtime=_runtime(
            api_key=api_key,
            client=client,
            default_params=default_params,
            timeout=timeout,
            mode=mode,
            result_limit=result_limit,
        ),
        include_examples=include_examples,
    )
    return as_provider_tool(definition, provider)


def shopping_search(
    *,
    provider: ProviderName | str = "auto",
    allowed_engines: Iterable[ShoppingSearchEngine | str]
    | ShoppingSearchEngine
    | str
    | None = None,
    default_engine: ShoppingSearchEngine | str | None = None,
    include_examples: bool = True,
    api_key: str | None = None,
    client: SearchClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    result_limit: int | None = 60,
    name: str = "shopping_search",
) -> Any:
    """Create a multi-marketplace product search tool.

    Parameters
    ----------
    provider
        Agent SDK adapter name, ``"auto"``, or ``"function"`` for a callable.
    allowed_engines
        Marketplaces exposed in the model-facing ``engine`` enum.
    default_engine
        Marketplace used when the model omits ``engine``.
    include_examples
        Add a short call example to the model-facing description.
    api_key, client, default_params, timeout, mode
        Runtime authentication, custom client, application filters, timeout, and
        compact or full response selection.
    result_limit
        Maximum items kept in each result list in both response modes; use
        ``None`` to keep all returned results.
    name
        Tool name presented to the model.

    Returns
    -------
    Any
        A provider-native tool whose input fields are ``query`` and ``engine``.
    """

    definition = _multi_engine_query_definition(
        name=name,
        description=(
            "Search products, prices, and merchants and return the SerpApi response as JSON."
        ),
        catalog=ShoppingSearchEngine,
        query_params=SHOPPING_QUERY_PARAM_BY_ENGINE,
        allowed_engines=allowed_engines,
        default_engine=default_engine,
        preferred_engine=ShoppingSearchEngine.GOOGLE_SHOPPING.value,
        engine_purposes={
            "google_shopping": "merchant comparison",
            "amazon": "listings",
            "walmart": "listings",
            "ebay": "listings",
        },
        runtime=_runtime(
            api_key=api_key,
            client=client,
            default_params=default_params,
            timeout=timeout,
            mode=mode,
            result_limit=result_limit,
        ),
        include_examples=include_examples,
    )
    return as_provider_tool(definition, provider)


def _fixed_query_factory(
    *,
    provider: ProviderName | str,
    engine: str,
    query_param: str,
    description: str,
    include_examples: bool,
    api_key: str | None,
    client: SearchClient | None,
    default_params: Mapping[str, Any] | None,
    timeout: float | None,
    mode: SearchResultMode | str,
    result_limit: int | None,
    name: str,
    validator: Any = None,
) -> Any:
    definition = _fixed_query_definition(
        name=name,
        engine=engine,
        query_param=query_param,
        description=description,
        runtime=_runtime(
            api_key=api_key,
            client=client,
            default_params=default_params,
            timeout=timeout,
            mode=mode,
            result_limit=result_limit,
        ),
        include_examples=include_examples,
        validator=validator,
    )
    return as_provider_tool(definition, provider)


def _validate_news_query_mode(params: Mapping[str, Any]) -> None:
    conflicts = {
        "topic_token",
        "kgmid",
        "publication_token",
        "section_token",
        "story_token",
        "so",
    }.intersection(params)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(
            "news_search supports Google News query mode; q cannot be combined with "
            f"advanced token parameters: {names}."
        )


def _validate_images_query_mode(params: Mapping[str, Any]) -> None:
    if {"location", "uule"}.issubset(params):
        raise ValueError("Google Images location and uule cannot be combined.")
    if {"period_unit", "period_value"}.intersection(params) and {
        "start_date",
        "end_date",
    }.intersection(params):
        raise ValueError(
            "Google Images relative period filters cannot be combined with start_date or end_date."
        )


def _validate_engine_mode(engine: str, params: Mapping[str, Any]) -> None:
    if engine == "google":
        if "location" in params and {"uule", "lat", "lon"}.intersection(params):
            raise ValueError("Google location cannot be combined with uule, lat, or lon.")
        if ("lat" in params) != ("lon" in params):
            raise ValueError("Google lat and lon must be supplied together.")
    elif engine == "google_light" and {"location", "uule"}.issubset(params):
        raise ValueError("Google Light location and uule cannot be combined.")
    elif engine == "duckduckgo":
        if {"search_assist", "m"}.issubset(params):
            raise ValueError("DuckDuckGo search_assist and m cannot be combined.")
        query = params.get("q")
        if isinstance(query, str) and len(query) > 500:
            raise ValueError("DuckDuckGo query must be 500 characters or fewer.")
    elif engine == "google_shopping" and {"location", "uule"}.issubset(params):
        raise ValueError("Google Shopping location and uule cannot be combined.")
    elif engine == "amazon" and "node" in params:
        raise ValueError("shopping_search Amazon keyword mode cannot be combined with node.")


def news_search(
    *,
    provider: ProviderName | str = "auto",
    include_examples: bool = True,
    api_key: str | None = None,
    client: SearchClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    result_limit: int | None = 20,
    name: str = "news_search",
) -> Any:
    """Create a Google News query-mode tool for current articles and stories.

    Parameters
    ----------
    provider
        Agent SDK adapter name, ``"auto"``, or ``"function"``.
    include_examples
        Add a short call example to the model-facing description.
    api_key, client, default_params, timeout, mode
        Runtime authentication, custom client, application filters, timeout, and
        compact or full response selection.
    result_limit
        Maximum items kept in each result list in both response modes; use
        ``None`` to keep all returned results.
    name
        Tool name presented to the model.

    Returns
    -------
    Any
        A provider-native tool with one required ``query`` input.
    """

    return _fixed_query_factory(
        provider=provider,
        engine="google_news",
        query_param="q",
        description="Search current news articles with Google News query mode.",
        include_examples=include_examples,
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
        result_limit=result_limit,
        name=name,
        validator=_validate_news_query_mode,
    )


def images_search(
    *,
    provider: ProviderName | str = "auto",
    include_examples: bool = True,
    api_key: str | None = None,
    client: SearchClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    result_limit: int | None = 50,
    name: str = "images_search",
) -> Any:
    """Create a Google Images tool for image URLs and metadata.

    Parameters
    ----------
    provider
        Agent SDK adapter name, ``"auto"``, or ``"function"``.
    include_examples
        Add a short call example to the model-facing description.
    api_key, client, default_params, timeout, mode
        Runtime authentication, custom client, application filters, timeout, and
        compact or full response selection.
    result_limit
        Maximum items kept in each result list in both response modes; use
        ``None`` to keep all returned results.
    name
        Tool name presented to the model.

    Returns
    -------
    Any
        A provider-native tool with one required ``query`` input.
    """

    return _fixed_query_factory(
        provider=provider,
        engine="google_images",
        query_param="q",
        description="Search Google Images for image URLs and metadata.",
        include_examples=include_examples,
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
        result_limit=result_limit,
        name=name,
        validator=_validate_images_query_mode,
    )


def videos_search(
    *,
    provider: ProviderName | str = "auto",
    include_examples: bool = True,
    api_key: str | None = None,
    client: SearchClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    result_limit: int | None = 10,
    name: str = "videos_search",
) -> Any:
    """Create a YouTube tool for videos, channels, and playlists.

    Parameters
    ----------
    provider
        Agent SDK adapter name, ``"auto"``, or ``"function"``.
    include_examples
        Add a short call example to the model-facing description.
    api_key, client, default_params, timeout, mode
        Runtime authentication, custom client, application filters, timeout, and
        compact or full response selection.
    result_limit
        Maximum items kept in each result list in both response modes; use
        ``None`` to keep all returned results.
    name
        Tool name presented to the model.

    Returns
    -------
    Any
        A provider-native tool with one required ``query`` input.
    """

    return _fixed_query_factory(
        provider=provider,
        engine="youtube",
        query_param="search_query",
        description="Search YouTube for videos, channels, and playlists.",
        include_examples=include_examples,
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
        result_limit=result_limit,
        name=name,
    )


def maps_search(
    *,
    provider: ProviderName | str = "auto",
    include_examples: bool = True,
    api_key: str | None = None,
    client: SearchClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    result_limit: int | None = 10,
    name: str = "maps_search",
) -> Any:
    """Create a Google Maps search-mode tool for places and businesses.

    The returned tool exposes ``query`` plus optional ``location``, ``zoom``,
    and ``nearby`` fields. ``zoom`` is constrained to 3 through 30, and
    ``nearby=True`` requires ``location``.

    Parameters
    ----------
    provider
        Agent SDK adapter name, ``"auto"``, or ``"function"``.
    include_examples
        Add a short call example to the model-facing description.
    api_key, client, default_params, timeout, mode
        Runtime authentication, custom client, application filters, timeout, and
        compact or full response selection.
    result_limit
        Maximum items kept in each result list in both response modes; use
        ``None`` to keep all returned results.
    name
        Tool name presented to the model.

    Returns
    -------
    Any
        A provider-native Google Maps search tool.
    """

    runtime = _runtime(
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
        result_limit=result_limit,
    )

    def maps_tool(
        query: str,
        location: str | None = None,
        zoom: int = 14,
        nearby: bool = False,
    ) -> str:
        if nearby and (not isinstance(location, str) or not location.strip()):
            raise ValueError("nearby=True requires location.")
        if isinstance(zoom, bool) or not isinstance(zoom, int) or not 3 <= zoom <= 30:
            raise ValueError("zoom must be an integer from 3 to 30.")
        typed_params: dict[str, Any] = {
            "q": query,
            "type": "search",
            "location": None,
            "z": None,
            "nearby": None,
        }
        if location is not None:
            if not location.strip():
                raise ValueError("location must not be empty when provided.")
            typed_params.update({"location": location.strip(), "z": zoom})
            if nearby:
                typed_params["nearby"] = True

        def validate(params: dict[str, Any]) -> None:
            require_nonempty(params, "q", label="query")
            if "place_id" in params or "data_cid" in params:
                raise ValueError(
                    "maps_search supports search mode; place_id and data_cid require a "
                    "separate place-details tool."
                )
            if "location" in params and {"ll", "lat", "lon"}.intersection(params):
                raise ValueError("Google Maps location and ll, lat, or lon cannot be combined.")
            if ("lat" in params) != ("lon" in params):
                raise ValueError("Google Maps lat and lon must be supplied together.")
            if {"open_state", "open_on_day"}.issubset(params):
                raise ValueError("Google Maps open_state and open_on_day cannot be combined.")

        return runtime.execute(
            engine="google_maps",
            typed_params=typed_params,
            validator=validate,
        )

    description = "Search Google Maps for places, businesses, and local discovery."
    if include_examples:
        description += " Example: query='coffee', location='Austin, Texas', zoom=14."
    schema = object_schema(
        {
            "query": {
                "type": "string",
                "description": (
                    "Place, business, or category; use location for a separate search origin."
                ),
            },
            "location": {
                "type": "string",
                "description": (
                    "Geographic search origin, e.g. 'Austin, Texas'; omit when query names "
                    "the area."
                ),
            },
            "zoom": {
                "type": "integer",
                "minimum": 3,
                "maximum": 30,
                "default": 14,
                "description": (
                    "Map zoom with location; 3 is broad and larger values are narrower."
                ),
            },
            "nearby": {
                "type": "boolean",
                "default": False,
                "description": (
                    "True for 'near me'; requires location. Leave false when query names the area."
                ),
            },
        },
        required=["query"],
    )
    maps_tool.__annotations__ = {
        "query": str,
        "location": str | None,
        "zoom": int,
        "nearby": bool,
        "return": str,
    }
    return as_provider_tool(ToolDefinition(maps_tool, name, description, schema), provider)
