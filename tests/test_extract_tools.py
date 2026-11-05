import asyncio
import inspect
import json
import sys
from copy import deepcopy
from types import ModuleType

import pytest

from serpapi_search_tools import SerpApiSearchError, web_crawl, web_fetch, web_map, web_search

PAGE_MARKDOWN = (
    "---\ntitle: Page frontmatter\n---\n\n# Café 東京\n\n"
    "## Error\n\nThis heading is page content.\n\n"
    "| A | B |\n| - | - |\n| 1 | 2 |\n\n```python\nprint('hello')\n```\n"
)
FULL_MARKDOWN = PAGE_MARKDOWN + "\n"
PAYLOAD = {
    "search_metadata": {"status": "Success"},
    "page_information": {"title": "Café 東京"},
    "markdown": PAGE_MARKDOWN,
    "inline_links": [{"link": "https://example.com/next"}],
}


class ExtractClient:
    def __init__(self, payload=None):
        self.calls = []
        self.payload = deepcopy(PAYLOAD if payload is None else payload)

    def extract(self, params):
        self.calls.append(dict(params))
        if params.get("output") == "md":
            return FULL_MARKDOWN
        return self.payload

    def search(self, params):
        raise AssertionError("extract must never call search")


def test_default_markdown_is_returned_unchanged_without_restriction():
    client = ExtractClient()
    tool = web_fetch(provider="function", client=client)

    assert tool(url="https://example.com") == FULL_MARKDOWN
    assert client.calls == [
        {
            "url": "https://example.com",
            "engine": "extract",
            "mode": "scrape",
            "output": "md",
        }
    ]
    assert client.payload == PAYLOAD


@pytest.mark.parametrize("mode", ["compact", "full"])
@pytest.mark.parametrize("response_format", ["markdown", "json"])
def test_response_modes_and_formats(mode, response_format):
    client = ExtractClient()
    tool = web_fetch(
        provider="function",
        client=client,
        mode=mode,
        response_format=response_format,
    )
    result = tool("https://example.com/path?a=1&b=two#section")
    assert client.calls[0]["url"] == "https://example.com/path?a=1&b=two#section"
    if response_format == "json":
        expected = {"markdown": PAGE_MARKDOWN} if mode == "compact" else PAYLOAD
        assert json.loads(result) == expected
        assert "Café 東京" in result
    else:
        assert result == FULL_MARKDOWN
    assert ("json_restrictor" in client.calls[0]) == (
        mode == "compact" and response_format == "json"
    )
    assert client.calls[0]["output"] == ("md" if response_format == "markdown" else "json")


def test_search_only_clients_are_not_used_as_a_fallback():
    calls = []

    class SearchOnlyClient:
        def search(self, params):
            calls.append(params)
            return PAYLOAD

    with pytest.raises(SerpApiSearchError, match="Custom extract client request failed"):
        web_fetch(provider="function", client=SearchOnlyClient())("https://example.com")
    assert calls == []


def test_extract_only_client_does_not_need_search():
    class ExtractOnlyClient:
        def extract(self, params):
            return PAGE_MARKDOWN

    assert (
        web_fetch(provider="function", client=ExtractOnlyClient())("https://example.com")
        == PAGE_MARKDOWN
    )


@pytest.mark.parametrize("result", ["", " \n", "\t\n\n"])
def test_markdown_preserves_empty_content(result):
    class Client:
        def extract(self, params):
            return result

    assert web_fetch(provider="function", client=Client())("https://example.com") == result


def test_markdown_without_api_frontmatter_preserves_all_sections():
    content = "# Page\n\n## Error\n\nPage content.\n\n## Links\n\n[Next](https://example.com/next)"

    class Client:
        def extract(self, params):
            return "\n" + content + "\n\n"

    assert (
        web_fetch(provider="function", client=Client())("https://example.com")
        == "\n" + content + "\n\n"
    )


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_redirected_engine_markdown_is_returned_unchanged(mode):
    result = (
        "---\nsearch_metadata:\n  status: Success\nsearch_parameters:\n  engine: google\n---\n"
        "\n# Coffee\n\n## Organic results\n\n[Coffee](https://example.com/coffee)\n\n"
    )

    class Client:
        def extract(self, params):
            assert "json_restrictor" not in params
            return result

    assert (
        web_fetch(provider="function", client=Client(), mode=mode)(
            "https://www.google.com/search?q=coffee"
        )
        == result
    )


def test_failed_extract_is_not_retried_as_search_and_hides_custom_credentials():
    class FailingClient(ExtractClient):
        def extract(self, params):
            raise RuntimeError("request failed: api_key=secret")

    with pytest.raises(SerpApiSearchError, match=r"^Custom extract client request failed\.$"):
        web_fetch(provider="function", client=FailingClient())("https://example.com")


@pytest.mark.parametrize(
    "url",
    [
        "",
        "  ",
        None,
        123,
        "example.com",
        "ftp://example.com",
        "https:///path",
        "https://[bad",
        "https://exa mple.com",
        "https://example.com/\npath",
    ],
)
def test_invalid_urls_fail_before_request(url):
    client = ExtractClient()
    tool = web_fetch(provider="function", client=client)
    with pytest.raises(ValueError, match="url"):
        tool(url)
    assert client.calls == []


@pytest.mark.parametrize("response_format", ["markdown", "json"])
def test_defaults_are_copied_and_only_compact_json_overrides_content_fields(response_format):
    defaults = {
        "include_tags": "main",
        "exclude_tags": "nav,footer",
        "timeout": 5000,
        "include_markdown": False,
        "json_restrictor": "page_information",
    }
    client = ExtractClient()
    tool = web_fetch(
        provider="function", client=client, default_params=defaults, response_format=response_format
    )
    defaults["include_tags"] = "article"
    tool("https://example.com")
    assert client.calls[0]["include_tags"] == "main"
    assert client.calls[0]["exclude_tags"] == "nav,footer"
    assert client.calls[0]["timeout"] == 5000
    assert client.calls[0]["include_markdown"] is (response_format == "json")
    assert client.calls[0]["json_restrictor"] == (
        "markdown,error" if response_format == "json" else "page_information"
    )
    assert defaults["include_markdown"] is False


@pytest.mark.parametrize("mode", ["crawl", "map"])
def test_only_single_page_scraping_is_supported(mode):
    with pytest.raises(ValueError, match="only mode='scrape'"):
        web_fetch(default_params={"mode": mode})


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"markdown": None},
        {"markdown": ""},
        {"markdown": "  "},
        {"markdown": []},
        {"organic_results": [{"title": "Routed to search"}]},
    ],
)
def test_missing_content_or_routed_search_does_not_report_success(payload):
    with pytest.raises(SerpApiSearchError, match=r"no markdown"):
        web_fetch(provider="function", client=ExtractClient(payload), response_format="json")(
            "https://example.com"
        )


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_api_errors_survive_restriction_and_hide_custom_secrets(mode):
    client = ExtractClient(
        {"error": "failed api_key=secret", "search_metadata": {"status": "Success"}}
    )
    with pytest.raises(
        SerpApiSearchError,
        match=r"^SerpApi extraction failed: The custom client returned an API error\.$",
    ):
        web_fetch(provider="function", client=client, mode=mode, response_format="json")(
            "https://example.com"
        )


@pytest.mark.parametrize("response_format", ["markdown", "json"])
def test_builtin_client_uses_sdk_extract_transport_and_reuses_client(monkeypatch, response_format):
    import serpapi
    from requests import Response

    calls = []

    def request(self, method, path, *, params, **kwargs):
        calls.append((self, method, path, dict(params), kwargs))
        response = Response()
        response.status_code = 200
        response.encoding = "utf-8"
        if params["output"] == "md":
            response.headers["Content-Type"] = "text/markdown; charset=utf-8"
            response._content = PAGE_MARKDOWN.encode()
        else:
            response.headers["Content-Type"] = "application/json"
            response._content = json.dumps({"markdown": PAGE_MARKDOWN}).encode()
        return response

    monkeypatch.setattr(serpapi.Client, "request", request)
    tool = web_fetch(
        provider="function",
        api_key="test-key",
        timeout=2.5,
        default_params={"timeout": 5000},
        response_format=response_format,
    )
    for url in ["https://example.com", "https://example.org"]:
        result = tool(url)
        if response_format == "markdown":
            assert result == PAGE_MARKDOWN
        else:
            assert json.loads(result) == {"markdown": PAGE_MARKDOWN}
    assert calls[0][0] is calls[1][0]
    assert calls[0][0].api_key == "test-key"
    assert calls[0][0].timeout == 2.5
    assert calls[0][1:3] == ("GET", "/search")
    expected_params = {
        "timeout": 5000,
        "url": "https://example.com",
        "engine": "extract",
        "mode": "scrape",
        "output": "md" if response_format == "markdown" else "json",
    }
    if response_format == "json":
        expected_params.update(include_markdown=True, json_restrictor="markdown,error")
    assert calls[0][3] == expected_params
    assert calls[0][4] == {}


@pytest.mark.parametrize("factory", [web_search, web_fetch, web_map, web_crawl])
def test_builtin_client_rotates_keys_and_redacts_in_flight_errors(monkeypatch, factory):
    from requests import Request, Response, Session

    keys_sent = []

    def request(self, method, url, params, **kwargs):
        keys_sent.append(params["api_key"])
        response = Response()
        response.url = Request(method, url, params=params).prepare().url
        if len(keys_sent) == 1:
            response.status_code = 200
            response.headers["Content-Type"] = "text/markdown"
            response._content = b"# Page"
        else:
            monkeypatch.setenv("SERPAPI_API_KEY", "third-test-key")
            response.status_code = 503
            response._content = b"Unavailable"
        return response

    monkeypatch.setattr(Session, "request", request)
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "first-test-key")
    tool = factory(provider="function")
    tool("https://example.com")
    monkeypatch.setenv("SERPAPI_API_KEY", "second-test-key")
    with pytest.raises(SerpApiSearchError) as failure:
        tool("https://example.com")
    assert keys_sent == ["first-test-key", "second-test-key"]
    assert "api_key=[REDACTED]" in str(failure.value)
    assert all(key not in str(failure.value) for key in keys_sent)

    monkeypatch.delenv("SERPAPI_API_KEY")
    with pytest.raises(RuntimeError, match="Set SERPAPI_API_KEY"):
        tool("https://example.com")
    assert len(keys_sent) == 2


def test_builtin_payload_error_is_sanitized(monkeypatch):
    import serpapi

    def reject(client, params):
        monkeypatch.setenv("SERPAPI_API_KEY", "replacement-test-key")
        return {"error": f"rejected api_key={client.api_key}"}

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(serpapi.Client, "extract", reject)
    with pytest.raises(SerpApiSearchError, match=r"rejected api_key=\[REDACTED\]"):
        web_fetch(provider="function")("https://example.com")


def test_openai_adapter_schema_and_async_invocation(monkeypatch):
    module = ModuleType("agents")

    class FunctionTool:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module.FunctionTool = FunctionTool
    monkeypatch.setitem(sys.modules, "agents", module)
    tool = web_fetch(provider="openai-agents", client=ExtractClient(), name="read_page")
    assert tool.name == "read_page"
    assert tool.params_json_schema["required"] == ["url"]
    assert set(tool.params_json_schema["properties"]) == {"url"}
    assert asyncio.run(tool.on_invoke_tool(None, '{"url":"https://example.com"}')) == FULL_MARKDOWN


def test_callable_metadata_and_engine_hint():
    tool = web_fetch(provider="function", include_examples=False)
    assert list(inspect.signature(tool).parameters) == ["url"]
    assert tool.__name__ == "web_fetch"
    assert "Example:" not in tool.__doc__
    with pytest.raises(ValueError, match=r"Use web_fetch\(\) instead"):
        web_search(allowed_engines=["extract"])


@pytest.mark.parametrize(
    "factory,operation,key,items,compact_items,restrictor,settings",
    [
        (
            web_map,
            "map",
            "urls",
            [{"url": "https://example.com/a", "source": "sitemap", "extra": 1}],
            [{"url": "https://example.com/a", "source": "sitemap"}],
            "urls[].{url,title,source},error,search_parameters.engine,search_metadata.status",
            {"include_sitemap": False},
        ),
        (
            web_crawl,
            "crawl",
            "pages",
            [
                {"url": "https://example.com/a", "markdown": "Café", "extra": 1},
                {"url": "https://example.com/b", "depth": 1, "error": "Page unavailable"},
            ],
            [
                {"url": "https://example.com/a", "markdown": "Café"},
                {"url": "https://example.com/b", "depth": 1, "error": "Page unavailable"},
            ],
            "pages[].{url,depth,markdown,error},error",
            {"max_depth": 0, "max_pages": 1},
        ),
    ],
)
def test_map_and_crawl_select_their_operation_and_output(
    factory, operation, key, items, compact_items, restrictor, settings
):
    payload = {key: items, "search_metadata": {"status": "Success"}}
    client = ExtractClient(payload)
    url = "https://example.com/"
    for mode in ("compact", "full"):
        for fmt in ("markdown", "json"):
            tool = factory(
                provider="function",
                client=client,
                mode=mode,
                response_format=fmt,
                default_params=settings,
            )
            assert list(inspect.signature(tool).parameters) == ["url"]
            result = tool(url)
            expected = dict(
                settings,
                url=url,
                mode=operation,
                engine="extract",
                output="md" if fmt == "markdown" else "json",
            )
            if fmt == "markdown":
                assert result == FULL_MARKDOWN
            elif mode == "full":
                assert json.loads(result) == payload
            else:
                expected["json_restrictor"] = restrictor
                if operation == "crawl":
                    expected["include_markdown"] = True
                assert json.loads(result) == {key: compact_items}
            assert client.calls[-1] == expected


@pytest.mark.parametrize("factory,key", [(web_map, "urls"), (web_crawl, "pages")])
def test_empty_collections_are_valid_and_redirects_remain_available_in_full_json(factory, key):
    client = ExtractClient({key: []})
    compact = factory(provider="function", client=client, response_format="json")
    assert json.loads(compact("https://example.com")) == {key: []}
    client.payload = {"organic_results": [{"title": "coffee"}]}
    with pytest.raises(SerpApiSearchError, match="redirected engine"):
        compact("https://www.google.com/search?q=coffee")
    full = factory(provider="function", client=client, response_format="json", mode="full")
    assert json.loads(full("https://www.google.com/search?q=coffee")) == client.payload


def test_compact_map_distinguishes_empty_success_from_redirect_and_invalid_response():
    client = ExtractClient(
        {"search_metadata": {"status": "Success"}, "search_parameters": {"engine": "extract"}}
    )
    compact = web_map(provider="function", client=client, response_format="json")
    assert json.loads(compact("https://example.com")) == {"urls": []}
    client.payload["search_parameters"]["engine"] = "google"
    with pytest.raises(SerpApiSearchError, match="redirected engine"):
        compact("https://www.google.com/search?q=coffee")
    client.payload["search_parameters"]["engine"] = "extract"
    client.payload["urls"] = None
    with pytest.raises(SerpApiSearchError):
        compact("https://example.com")


@pytest.mark.parametrize(
    "factory,settings",
    [
        (web_map, {"mode": "crawl"}),
        (web_crawl, {"mode": "map"}),
        (web_crawl, {"max_pages": 0}),
        (web_crawl, {"max_pages": 101}),
        (web_crawl, {"max_depth": 4}),
        (web_crawl, {"max_depth": -1}),
        (web_crawl, {"max_pages": True}),
    ],
)
def test_crawl_limits_and_operation_are_fixed_before_any_requests(factory, settings):
    with pytest.raises(ValueError, match=next(iter(settings))):
        factory(provider="function", default_params=settings)
