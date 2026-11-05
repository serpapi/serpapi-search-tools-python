from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

from serpapi_search_tools._adapters import as_provider_tool
from serpapi_search_tools._shared import (
    ExtractClient,
    ProviderName,
    SearchResultFormat,
    SearchResultMode,
    SearchRuntime,
    SerpApiSearchError,
    ToolDefinition,
    object_schema,
    require_nonempty,
)

_Operation = Literal["scrape", "map", "crawl"]
_DESCRIPTIONS = {
    "scrape": "Extract the content of a public web page from its absolute http(s) URL.",
    "map": "Discover URLs on a public website from its links and sitemaps.",
    "crawl": "Follow links on a public website and read multiple pages. Each successfully "
    "extracted page costs one search credit.",
}
_COLLECTIONS = {
    "map": ("urls", ("url", "title", "source")),
    "crawl": ("pages", ("url", "depth", "markdown", "error")),
}


class ExtractRuntime(SearchRuntime):
    """Run Extract API operations through the shared client runtime."""

    def extract(self, url: str, operation: _Operation) -> str:
        url = require_nonempty({"url": url}, "url", label="url")
        try:
            parsed = urlsplit(url)
            valid_url = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
        except ValueError:
            valid_url = False
        if not valid_url or any(character.isspace() for character in url):
            raise ValueError("url must be an absolute http(s) URL without whitespace.")

        params = dict(self.default_params or {})
        markdown = self.response_format is SearchResultFormat.MARKDOWN
        compact_json = not markdown and self.mode is SearchResultMode.COMPACT
        params.update(engine="extract", url=url, mode=operation)
        if compact_json:
            if operation != "map":
                params["include_markdown"] = True
            if operation == "scrape":
                params["json_restrictor"] = "markdown,error"
            else:
                key, fields = _COLLECTIONS[operation]
                params["json_restrictor"] = f"{key}[].{{{','.join(fields)}}},error"
                if operation == "map":
                    params["json_restrictor"] += ",search_parameters.engine,search_metadata.status"
        params["output"] = "md" if markdown else "json"
        result = self._request(params, extract=True)

        if not markdown and isinstance(result, Mapping):
            payload = _compact_extract_result(result, operation) if compact_json else dict(result)
            return json.dumps(payload, default=str, ensure_ascii=False, separators=(",", ":"))
        if markdown and isinstance(result, str):
            return result
        expected = "str" if markdown else "a mapping"
        raise TypeError(f"Extract client must return {expected} for output={params['output']!r}.")


def _compact_extract_result(result: Mapping[str, Any], operation: _Operation) -> dict[str, Any]:
    if operation == "scrape":
        markdown = result.get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            raise SerpApiSearchError(
                "SerpApi extraction returned no markdown. "
                "The page may have no extractable content or use a dedicated search engine."
            )
        return {"markdown": markdown}

    key, fields = _COLLECTIONS[operation]
    if operation == "map" and key not in result:
        match result:
            case {
                "search_parameters": {"engine": "extract"},
                "search_metadata": {"status": "Success"},
            }:
                return {key: []}
    items = result.get(key)
    if not isinstance(items, list):
        raise SerpApiSearchError(
            f"Expected a list of {key} from Extract {operation}. "
            "Use full JSON to inspect a redirected engine response."
        )
    selected = []
    for item in items:
        if not isinstance(item, Mapping):
            raise SerpApiSearchError(f"Expected an object in Extract {operation}'s {key} list.")
        selected.append({field: item[field] for field in fields if field in item})
    return {key: selected}


def _extract_tool(
    operation: _Operation,
    *,
    provider: ProviderName | str,
    include_examples: bool,
    api_key: str | None,
    client: ExtractClient | None,
    default_params: Mapping[str, Any] | None,
    timeout: float | None,
    mode: SearchResultMode | str,
    response_format: SearchResultFormat | str,
    name: str,
) -> Any:
    tool_name = "web_fetch" if operation == "scrape" else f"web_{operation}"
    if (default_params or {}).get("mode", operation) != operation:
        raise ValueError(f"{tool_name} supports only mode={operation!r} in default_params.")
    if operation == "crawl":
        for field, lower, upper in (("max_pages", 1, 100), ("max_depth", 0, 3)):
            if default_params is not None and field in default_params:
                value = default_params[field]
                if type(value) is not int or not lower <= value <= upper:
                    raise ValueError(f"{field} must be an integer from {lower} to {upper}.")
    runtime = ExtractRuntime(
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
        response_format=response_format,
        result_limit=None,
    )

    def extract(url: str) -> str:
        return runtime.extract(url, operation)

    description = _DESCRIPTIONS[operation]
    if include_examples:
        description += " Example: url='https://example.com'."
    schema = object_schema(
        {"url": {"type": "string", "description": "The absolute http(s) URL to extract."}},
        required=["url"],
    )
    return as_provider_tool(ToolDefinition(extract, name, description, schema), provider)


def web_fetch(
    *,
    provider: ProviderName | str = "auto",
    include_examples: bool = True,
    api_key: str | None = None,
    client: ExtractClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    response_format: SearchResultFormat | str = SearchResultFormat.MARKDOWN,
    name: str = "web_fetch",
) -> Any:
    """Create a tool that extracts content from one public web page.

    Parameters
    ----------
    provider
        Agent SDK adapter name, ``"auto"``, or ``"function"`` for a callable.
    include_examples
        Add a URL example to the model-facing description.
    api_key, client, timeout
        Authentication, optional client, and transport timeout in seconds.
        Clients must implement ``extract(params)``.
    default_params
        Extract API settings such as ``include_tags`` and ``exclude_tags``.
        This tool supports only ``mode="scrape"``.
    mode
        For JSON, ``"compact"`` selects the top-level ``markdown`` field and
        ``"full"`` returns the complete response. Markdown is always
        returned unchanged.
    response_format
        ``"markdown"`` (default) requests ``output="md"``. ``"json"``
        requests ``output="json"``.
    name
        Tool name presented to the model.

    Returns
    -------
    Any
        A provider-native tool with one required ``url`` input.
    """
    return _extract_tool(
        "scrape",
        provider=provider,
        include_examples=include_examples,
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
        response_format=response_format,
        name=name,
    )


def web_map(
    *,
    provider: ProviderName | str = "auto",
    include_examples: bool = True,
    api_key: str | None = None,
    client: ExtractClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    response_format: SearchResultFormat | str = SearchResultFormat.MARKDOWN,
    name: str = "web_map",
) -> Any:
    """Discover a website's URLs using the Extract API's map operation.

    The agent supplies a single URL. Set ``include_sitemap`` and
    ``include_subdomains`` in ``default_params`` to control discovery.
    Markdown returns the API's link list. Compact JSON keeps the URL, title,
    and source of each link. Full JSON returns the entire API payload.
    """
    return _extract_tool(
        "map",
        provider=provider,
        include_examples=include_examples,
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
        response_format=response_format,
        name=name,
    )


def web_crawl(
    *,
    provider: ProviderName | str = "auto",
    include_examples: bool = True,
    api_key: str | None = None,
    client: ExtractClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    response_format: SearchResultFormat | str = SearchResultFormat.MARKDOWN,
    name: str = "web_crawl",
) -> Any:
    """Follow a website's links and extract multiple pages using the Extract API.

    The agent supplies a single URL. Set ``max_pages`` (1 to 100, default 10)
    and ``max_depth`` (0 to 3, default 2) in ``default_params``. Each successful
    page costs one search credit, including the starting page. Markdown
    returns the API body. Compact JSON keeps page URLs, depths, Markdown,
    and per-page errors. Full JSON returns the entire API payload.
    """
    return _extract_tool(
        "crawl",
        provider=provider,
        include_examples=include_examples,
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
        response_format=response_format,
        name=name,
    )
