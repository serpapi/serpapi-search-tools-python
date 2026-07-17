from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from threading import RLock
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
    "SerpApiSearchError",
    "detect_provider",
    "normalize_provider",
]

ToolFunction: TypeAlias = Callable[..., str]
_RESERVED_DEFAULT_PARAMS = frozenset({"api_key", "async", "engine", "output"})


class SerpApiSearchError(RuntimeError):
    """A sanitized failure raised by the built-in SerpApi client."""


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
    _builtin_client: SearchClient | None = field(init=False, default=None, repr=False)
    _builtin_client_lock: Any = field(init=False, default_factory=RLock, repr=False)

    def __post_init__(self) -> None:
        defaults = dict(self.default_params or {})
        reserved = sorted(_RESERVED_DEFAULT_PARAMS.intersection(defaults))
        if reserved:
            names = ", ".join(reserved)
            raise ValueError(f"default_params cannot set transport-controlled parameters: {names}.")
        self.default_params = defaults

    def execute(
        self,
        *,
        engine: str,
        typed_params: Mapping[str, Any],
        validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        params = dict(self.default_params or {})
        params.update({key: value for key, value in typed_params.items() if value is not None})
        params["engine"] = engine
        if validator is not None:
            validator(params)

        client = self._client()
        if self.client is None:
            try:
                with self._builtin_client_lock:
                    result = client.search(params)
            except Exception as exc:
                api_key = self.api_key or env_api_key()
                message = str(exc)
                if api_key:
                    message = message.replace(api_key, "[REDACTED]")
                raise SerpApiSearchError(f"SerpApi request failed: {message}") from None
        else:
            result = client.search(params)
        return json.dumps(dict(result), default=str, separators=(",", ":"))

    def _client(self) -> SearchClient:
        if self.client is not None:
            return self.client

        with self._builtin_client_lock:
            if self._builtin_client is not None:
                return self._builtin_client

            try:
                import serpapi
            except ImportError as exc:
                message = "serpapi is required. Install it with `pip install serpapi-search-tools`."
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
            self._builtin_client = cast(
                SearchClient,
                cast(Callable[..., Any], client_factory)(**kwargs),
            )
            return self._builtin_client


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
