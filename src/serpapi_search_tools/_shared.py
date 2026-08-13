from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from threading import RLock, local
from typing import Any, Protocol, TypeAlias, cast

from serpapi_search_tools._providers import (
    PROVIDER_ALIASES,
    ProviderName,
    detect_provider,
    normalize_provider,
)

__all__ = [
    "PROVIDER_ALIASES",
    "ProviderName",
    "SearchResultMode",
    "SerpApiSearchError",
    "detect_provider",
    "normalize_provider",
]

ToolFunction: TypeAlias = Callable[..., str]
_RESERVED_DEFAULT_PARAMS = frozenset({"api_key", "async", "engine", "output"})
_DEFAULT_RESULT_LIMIT = 10
_COMPACT_DROPPED_RESULT_KEYS_BY_ENGINE: Mapping[str, frozenset[str]] = {
    "google_images": frozenset(
        {
            "related_content_id",
            "serpapi_related_content_link",
            "source_logo",
        }
    ),
    "google_shopping": frozenset(
        {
            "immersive_product_page_token",
            "serpapi_immersive_product_api",
            "serpapi_thumbnail",
            "source_icon",
        }
    ),
    "amazon": frozenset(
        {
            "link_clean",
            "purchase_options",
            "serpapi_link",
        }
    ),
    "walmart": frozenset(
        {
            "muliple_options_available",
            "seller_id",
            "serpapi_product_page_url",
            "variant_swatches",
        }
    ),
    "ebay": frozenset({"buying_format_text", "serpapi_link", "watchers"}),
    "google_hotels": frozenset(
        {
            "nearby_places",
            "reviews_breakdown",
            "serpapi_google_hotels_photos_link",
            "serpapi_google_hotels_reviews_link",
            "serpapi_property_details_link",
        }
    ),
    "google_travel_explore": frozenset({"serpapi_link"}),
}
_GOOGLE_LIGHT_RESULT_KEYS = (
    "answer_box",
    "knowledge_graph",
    "organic_results",
    "related_questions",
    "related_searches",
    "top_stories",
)
_SAFE_SEARCH_INFORMATION_KEYS = (
    "query_displayed",
    "total_results",
    "time_taken_displayed",
    "organic_results_state",
)
_COMPACT_RESULT_KEYS_BY_ENGINE: Mapping[str, tuple[str, ...]] = {
    "google": ("answer_box", "knowledge_graph", "ai_overview", "organic_results"),
    "google_light": _GOOGLE_LIGHT_RESULT_KEYS,
    "bing": ("answer_box", "knowledge_graph", "copilot_answer", "organic_results"),
    "yahoo": ("answer_box", "knowledge_graph", "organic_results"),
    "duckduckgo": ("knowledge_graph", "organic_results"),
    "google_news": ("news_results",),
    "google_maps": ("local_results",),
    "google_images": ("images_results",),
    "google_shopping": ("shopping_results",),
    "amazon": ("organic_results",),
    "walmart": ("organic_results",),
    "ebay": ("organic_results",),
    "youtube": (
        "video_results",
        "shorts_results",
        "channel_results",
        "playlist_results",
        "movie_results",
        "category_results",
    ),
    "google_hotels": ("properties",),
    "google_flights": ("best_flights", "other_flights"),
    "google_travel_explore": ("destinations",),
}


class SerpApiSearchError(RuntimeError):
    """A sanitized failure raised while executing a search."""


class SearchResultMode(str, Enum):
    """Control how much of a SerpApi response is returned to the agent."""

    COMPACT = "compact"
    FULL = "full"


class SearchClient(Protocol):
    """Small client contract accepted by every search-tool factory."""

    def search(self, params: dict[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ToolDefinition:
    """Provider-neutral callable plus the metadata needed by SDK adapters."""

    function: ToolFunction
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class SearchRuntime:
    """Merge, validate, execute, and encode one SerpApi request."""

    api_key: str | None = field(default=None, repr=False)
    client: SearchClient | None = None
    default_params: Mapping[str, Any] | None = None
    timeout: float | None = None
    mode: SearchResultMode | str = SearchResultMode.COMPACT
    result_limit: int | None = _DEFAULT_RESULT_LIMIT
    _builtin_clients: Any = field(init=False, default_factory=local, repr=False)
    _builtin_client_lock: Any = field(init=False, default_factory=RLock, repr=False)

    def __post_init__(self) -> None:
        defaults = dict(self.default_params or {})
        reserved = sorted(_RESERVED_DEFAULT_PARAMS.intersection(defaults))
        if reserved:
            names = ", ".join(reserved)
            raise ValueError(f"default_params cannot set transport-controlled parameters: {names}.")
        self.default_params = defaults
        try:
            self.mode = SearchResultMode(self.mode)
        except ValueError as exc:
            choices = ", ".join(member.value for member in SearchResultMode)
            raise ValueError(f"mode must be one of: {choices}.") from exc
        if self.result_limit is not None:
            if isinstance(self.result_limit, bool) or not isinstance(self.result_limit, int):
                raise ValueError("result_limit must be a positive integer or None.")
            if self.result_limit < 1:
                raise ValueError("result_limit must be a positive integer or None.")

    def execute(
        self,
        *,
        engine: str,
        typed_params: Mapping[str, Any],
        validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        params = dict(self.default_params or {})
        for key, value in typed_params.items():
            if value is None:
                params.pop(key, None)
            else:
                params[key] = value
        params["engine"] = engine
        if validator is not None:
            validator(params)

        client = self._client()
        try:
            result = client.search(params)
        except Exception as exc:
            if self.client is None:
                api_key = self.api_key or env_api_key()
                message = _sanitized_provider_error(exc, api_key=api_key)
                raise SerpApiSearchError(f"SerpApi request failed: {message}") from None
            raise SerpApiSearchError("Custom search client request failed.") from None
        plain_result = dict(result)
        plain_result = _limit_result_lists(
            plain_result,
            engine=engine,
            result_limit=self.result_limit,
        )
        if self.mode is SearchResultMode.COMPACT:
            plain_result = _compact_result(plain_result, engine=engine)
        return json.dumps(plain_result, default=str, separators=(",", ":"))

    def _client(self) -> SearchClient:
        if self.client is not None:
            return self.client

        client = getattr(self._builtin_clients, "client", None)
        if client is not None:
            return cast(SearchClient, client)

        with self._builtin_client_lock:
            try:
                import serpapi
            except ImportError as exc:
                message = "serpapi is required. Install or repair it with `pip install serpapi`."
                raise ImportError(message) from exc

            api_key = self.api_key or env_api_key()
            if not api_key:
                raise RuntimeError(
                    "Set SERPAPI_API_KEY or SERPAPI_KEY, or pass api_key=..., before searching."
                )

            client_factory = getattr(serpapi, "Client", None)
            if client_factory is None:
                if getattr(serpapi, "SerpApiClient", None) is not None:
                    raise RuntimeError(
                        "The legacy google-search-results package is shadowing the supported "
                        "serpapi SDK. Uninstall google-search-results and reinstall serpapi."
                    )
                raise RuntimeError("The installed serpapi package does not expose Client.")

            kwargs: dict[str, Any] = {"api_key": api_key}
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout
            client = cast(
                SearchClient,
                cast(Callable[..., Any], client_factory)(**kwargs),
            )
            self._builtin_clients.client = client
            return client


def _sanitized_provider_error(exc: Exception, *, api_key: str | None) -> str:
    message = str(exc)
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            provider_error = payload.get("error")
            if isinstance(provider_error, str) and provider_error.strip():
                message = provider_error.strip()
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message


def _compact_result(
    result: Mapping[str, Any],
    *,
    engine: str,
) -> dict[str, Any]:
    compact = {"error": result["error"]} if "error" in result else {}
    included_result = False
    for key in _COMPACT_RESULT_KEYS_BY_ENGINE.get(engine, ()):
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, list):
            compact[key] = [_compact_result_item(item, engine=engine) for item in value]
        else:
            compact[key] = value
        included_result = True
    if "error" in compact or included_result:
        return compact

    search_information = result.get("search_information")
    if isinstance(search_information, Mapping):
        safe_information = {
            key: search_information[key]
            for key in _SAFE_SEARCH_INFORMATION_KEYS
            if key in search_information
        }
        if safe_information:
            compact["search_information"] = safe_information
    search_metadata = result.get("search_metadata")
    if isinstance(search_metadata, Mapping) and "status" in search_metadata:
        compact["search_metadata"] = {"status": search_metadata["status"]}
    compact["no_results"] = True
    return compact


def _compact_result_item(item: Any, *, engine: str) -> Any:
    if not isinstance(item, Mapping):
        return item

    dropped_keys = _COMPACT_DROPPED_RESULT_KEYS_BY_ENGINE.get(engine, frozenset())
    compact_item = {key: value for key, value in item.items() if key not in dropped_keys}

    if engine == "amazon":
        clean_link = item.get("link_clean")
        if isinstance(clean_link, str) and clean_link:
            compact_item["link"] = clean_link
    elif engine == "google_hotels":
        images = compact_item.get("images")
        if isinstance(images, list):
            compact_item["images"] = images[:1]

    return compact_item


def _limit_result_lists(
    result: Mapping[str, Any],
    *,
    engine: str,
    result_limit: int | None,
) -> dict[str, Any]:
    limited = dict(result)
    if result_limit is None:
        return limited
    for key in _COMPACT_RESULT_KEYS_BY_ENGINE.get(engine, ()):
        value = limited.get(key)
        if isinstance(value, list):
            limited[key] = value[:result_limit]
    return limited


def object_schema(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    required: Iterable[str],
) -> dict[str, Any]:
    """Build the strict object schema shared by explicit-schema adapters."""

    return {
        "type": "object",
        "properties": {name: dict(schema) for name, schema in properties.items()},
        "required": list(required),
        "additionalProperties": False,
    }


def require_nonempty(params: Mapping[str, Any], key: str, *, label: str | None = None) -> str:
    """Return a stripped required string or raise a field-specific error."""

    value = params.get(key)
    field = label or key
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty.")
    return value.strip()


def require_positive(value: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be at least 1.")


def require_nonnegative(value: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be 0 or greater.")


def parse_iso_date(value: str, *, field: str) -> date:
    message = f"{field} must be an ISO date in YYYY-MM-DD format."
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if parsed.isoformat() != value:
        raise ValueError(message)
    return parsed


def env_api_key() -> str | None:
    return os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
