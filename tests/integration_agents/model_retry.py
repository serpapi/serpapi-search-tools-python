from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import pytest


class RecordingClient(Protocol):
    calls: list[dict[str, Any]]


def run_until_tool_called(
    run_once: Callable[[], Any],
    search_client: RecordingClient,
    *,
    attempts: int = 3,
) -> Any:
    """Retry bounded local-model tool selection without hiding persistent failures."""

    results: list[Any] = []
    for _ in range(attempts):
        results.append(run_once())
        if search_client.calls:
            return results[-1]
    pytest.fail(
        f"The model did not call the supplied search tool after {attempts} attempts. "
        f"Last result: {results[-1]!r}"
    )
