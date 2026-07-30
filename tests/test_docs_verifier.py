from __future__ import annotations

from pathlib import Path

import pytest
from scripts.fix_built_docs_links import repair

from tests.verify_built_docs import verify


def _write_page(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")


def test_root_absolute_link_outside_project_prefix_is_rejected(tmp_path: Path) -> None:
    site = tmp_path / "_site"
    _write_page(site / "index.html", '<a href="/missing.html">broken</a>')

    with pytest.raises(AssertionError, match="outside the project site"):
        verify(site)


def test_external_https_link_is_not_treated_as_a_root_absolute_site_link(
    tmp_path: Path,
) -> None:
    site = tmp_path / "_site"
    _write_page(site / "index.html", '<a href="https://serpapi.com/search-api">docs</a>')

    verify(site)


def test_relative_link_cannot_escape_site_even_when_target_exists(tmp_path: Path) -> None:
    site = tmp_path / "_site"
    (tmp_path / "outside.txt").write_text("not deployed", encoding="utf-8")
    _write_page(site / "index.html", '<a href="../outside.txt">broken</a>')

    with pytest.raises(AssertionError, match="outside the rendered site"):
        verify(site)


def test_qualified_enum_member_links_are_repaired_to_rendered_anchors(
    tmp_path: Path,
) -> None:
    site = tmp_path / "_site"
    _write_page(
        site / "reference" / "SearchResultMode.html",
        '<h4 id="full">FULL</h4>',
    )
    _write_page(
        site / "user-guide" / "configuration.html",
        (
            '<a href="../reference/SearchResultMode.html'
            '#serpapi_search_tools.SearchResultMode.FULL">full</a>'
        ),
    )

    assert repair(site) == 1
    verify(site)
