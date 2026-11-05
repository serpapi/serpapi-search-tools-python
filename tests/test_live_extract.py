"""Opt-in coverage for the Extract API on a configured server."""

import json
import os

import pytest
import serpapi

from serpapi_search_tools import SerpApiSearchError, web_crawl, web_fetch, web_map

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def extract_endpoint(monkeypatch):
    base_url = os.getenv("SERPAPI_EXTRACT_BASE_URL", "https://serpapi.com")
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        pytest.skip("Set a SerpApi key accepted by the Extract server.")
    monkeypatch.setattr(serpapi.Client, "BASE_DOMAIN", base_url.rstrip("/"))


@pytest.mark.parametrize(
    "url,content_marker",
    [
        ("https://example.com", "documentation examples"),
        ("https://www.python.org/about/", "Python"),
    ],
)
def test_live_compact_content_matches_native_markdown_body(url, content_marker):
    full = json.loads(
        web_fetch(
            provider="function",
            mode="full",
            response_format="json",
            timeout=90,
        )(url)
    )
    assert full["search_metadata"]["status"] == "Success"
    assert full["page_information"]["title"]
    markdown = full["markdown"]
    assert content_marker in markdown
    assert markdown.strip()

    compact = web_fetch(provider="function", timeout=90)(url)
    native_markdown = web_fetch(provider="function", mode="full", timeout=90)(url)
    assert compact == native_markdown
    assert compact.strip() == markdown.strip()
    assert content_marker in compact
    assert "search_metadata:" not in compact

    compact_json = json.loads(
        web_fetch(
            provider="function",
            response_format="json",
            timeout=90,
        )(url)
    )
    assert compact_json == {"markdown": markdown}


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_live_markdown_content_filters(mode):
    result = web_fetch(
        provider="function",
        mode=mode,
        timeout=90,
        default_params={"include_tags": "h1"},
    )("https://www.python.org/about/")
    assert "Python is powerful" in result
    assert "Getting Started" not in result


def test_live_full_json_can_disable_default_markdown():
    result = web_fetch(
        provider="function",
        mode="full",
        response_format="json",
        timeout=90,
        default_params={"include_markdown": False},
    )("https://example.com")
    payload = json.loads(result)
    assert payload["search_metadata"]["status"] == "Success"
    assert payload["page_information"]["title"] == "Example Domain"
    assert "markdown" not in payload


def test_live_json_restrictor_filters_the_raw_sdk_response():
    client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY"))
    result = client.extract(
        {
            "url": "https://example.com",
            "mode": "scrape",
            "output": "json",
            "include_markdown": True,
            "json_restrictor": "markdown,error",
        },
        timeout=90,
    )
    assert set(result) == {"markdown"}
    assert "documentation examples" in result["markdown"]


def test_live_json_selector_without_content_raises():
    with pytest.raises(SerpApiSearchError, match="returned no"):
        web_fetch(
            provider="function",
            response_format="json",
            timeout=90,
            default_params={"include_tags": ".serpapi-nonexistent-selector"},
        )("https://example.com")


@pytest.mark.parametrize(
    "url",
    ["https://www.google.com/search?q=coffee", "https://www.bing.com/search?q=coffee"],
)
@pytest.mark.parametrize("response_format", ["markdown", "json"])
def test_live_redirected_engine_responses_are_preserved(url, response_format, monkeypatch):
    extract = serpapi.Client.extract
    responses = []

    def record_response(self, params, **kwargs):
        assert "json_restrictor" not in params
        result = extract(self, params, **kwargs)
        responses.append(result)
        return result

    monkeypatch.setattr(serpapi.Client, "extract", record_response)
    result = web_fetch(
        provider="function",
        response_format=response_format,
        mode="compact" if response_format == "markdown" else "full",
        timeout=90,
    )(url)
    if response_format == "markdown":
        assert result == responses[0]
        assert result.startswith("---\nsearch_metadata:")
        assert "coffee" in result.lower()
    else:
        payload = json.loads(result)
        assert payload == dict(responses[0])
        assert payload["search_parameters"]["engine"] in {"google", "bing"}
        assert payload["organic_results"]
        assert "markdown" not in payload


def test_live_compact_json_redirect_without_markdown_raises():
    with pytest.raises(SerpApiSearchError, match="dedicated search engine"):
        web_fetch(provider="function", response_format="json", timeout=90)(
            "https://www.google.com/search?q=coffee"
        )


def test_live_compact_map_distinguishes_an_empty_site_from_a_search_redirect():
    discover = web_map(
        provider="function",
        response_format="json",
        default_params={"include_sitemap": False},
        timeout=90,
    )
    assert json.loads(discover("https://example.com/")) == {"urls": []}
    with pytest.raises(SerpApiSearchError, match="redirected engine"):
        discover("https://www.google.com/search?q=coffee")


def test_live_markdown_preserves_unicode():
    result = web_fetch(provider="function", timeout=90)("https://fr.wikipedia.org/wiki/Caf%C3%A9")
    assert "café" in result.lower()
    assert "\ufffd" not in result


def test_live_api_rejection_raises_without_leaking_key():
    with pytest.raises(SerpApiSearchError) as failure:
        web_fetch(provider="function", timeout=90)("http://127.0.0.1/")
    key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    assert key not in str(failure.value)


def test_live_invalid_key_raises():
    with pytest.raises(SerpApiSearchError):
        web_fetch(provider="function", api_key="invalid-extract-key", timeout=30)(
            "https://example.com"
        )


@pytest.mark.parametrize(
    "factory,key,settings",
    [
        (web_map, "urls", {"include_sitemap": False}),
        (web_crawl, "pages", {"max_pages": 2, "max_depth": 1}),
    ],
)
@pytest.mark.parametrize(
    "fmt,mode", [("markdown", "compact"), ("json", "compact"), ("json", "full")]
)
def test_live_map_and_crawl(factory, key, settings, fmt, mode):
    url = "https://www.python.org/about/"
    result = factory(
        provider="function", response_format=fmt, mode=mode, default_params=settings, timeout=120
    )(url)
    if fmt == "markdown":
        assert "https://www.python.org/" in result
        return
    payload = json.loads(result)
    items = payload[key]
    assert items and all(item["url"] for item in items)
    if key == "pages":
        assert len(items) == 2
        assert all(item["depth"] <= 1 for item in items)
        assert any(item.get("markdown") for item in items)
    if mode == "compact":
        assert set(payload) == {key}
        fields = (
            {"url", "title", "source"} if key == "urls" else {"url", "depth", "markdown", "error"}
        )
        assert all(set(item) <= fields for item in items)
    else:
        assert payload["search_metadata"]["status"] == "Success"
