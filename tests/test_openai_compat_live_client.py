from __future__ import annotations

import httpx
import pytest

from tests.openai_compat_llm_tests.conftest import RecordingLiveSearchClient


def test_failed_live_response_is_not_recorded_as_a_successful_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = httpx.Response(
        401,
        request=httpx.Request("GET", "https://serpapi.com/search.json"),
    )
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: response)
    client = RecordingLiveSearchClient(api_key="revoked")

    with pytest.raises(httpx.HTTPStatusError):
        client.search({"engine": "google_light", "q": "coffee"})

    assert client.calls == []
