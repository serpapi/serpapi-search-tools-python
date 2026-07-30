"""Repair known Great Docs 0.14 API-reference link mismatches."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_OBJECT_FRAGMENT = re.compile(
    r'(?P<prefix>href="(?P<path>[^"#]*?(?P<name>[A-Za-z][A-Za-z0-9_]*)\.html))'
    r"#serpapi_search_tools\.(?P=name)\""
)
_QUALIFIED_ENUM_FRAGMENT = re.compile(
    r'(?P<prefix>href="[^"#]*?(?P<name>[A-Za-z][A-Za-z0-9_]*)\.html)'
    r"#serpapi_search_tools\.(?P=name)\.(?P<member>[A-Z][A-Z0-9_]*)\""
)
_ENUM_FRAGMENT = re.compile(r'href="#(?P<name>[A-Z][A-Z0-9_]*)"')
_SOURCE_PREFIX = (
    "https://github.com/serpapi/serpapi-search-tools-python/blob/main/serpapi_search_tools/"
)
_CORRECTED_SOURCE_PREFIX = (
    "https://github.com/serpapi/serpapi-search-tools-python/blob/main/src/serpapi_search_tools/"
)


def repair(site_root: Path) -> int:
    changed = 0
    for path in site_root.rglob("*.html"):
        source = path.read_text(encoding="utf-8")
        repaired = _QUALIFIED_ENUM_FRAGMENT.sub(
            lambda match: f'{match.group("prefix")}#{match.group("member").lower()}"',
            source,
        )
        repaired = _OBJECT_FRAGMENT.sub(r'\g<prefix>"', repaired)
        repaired = _ENUM_FRAGMENT.sub(
            lambda match: f'href="#{match.group("name").lower()}"',
            repaired,
        )
        repaired = repaired.replace(_SOURCE_PREFIX, _CORRECTED_SOURCE_PREFIX)
        if repaired != source:
            path.write_text(repaired, encoding="utf-8")
            changed += 1
    return changed


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("great-docs/_site")
    print(f"Repaired API-reference links in {repair(root)} rendered page(s).")
