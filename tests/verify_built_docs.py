"""Fail when rendered documentation contains a broken local link or anchor."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PREFIX = "https://github.com/serpapi/serpapi-search-tools-python/blob/main/"
SITE_PREFIX = "/serpapi-search-tools-python/"


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        anchor = attributes.get("id") or attributes.get("name")
        if anchor:
            self.anchors.add(anchor)
        if tag == "a" and attributes.get("href"):
            self.links.append(str(attributes["href"]))


def _load_pages(site_root: Path) -> dict[Path, _PageParser]:
    pages: dict[Path, _PageParser] = {}
    for path in site_root.rglob("*.html"):
        parser = _PageParser()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = parser
    return pages


def _local_target(site_root: Path, source: Path, href: str) -> tuple[Path, str] | None:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or href.startswith(("mailto:", "tel:", "javascript:")):
        return None

    path_text = unquote(parsed.path)
    if path_text.startswith(SITE_PREFIX):
        target = site_root / path_text.removeprefix(SITE_PREFIX)
    elif path_text.startswith("/"):
        return None
    elif path_text:
        target = source.parent / path_text
    else:
        target = source

    if target.is_dir() or path_text.endswith("/"):
        target = target / "index.html"
    return target.resolve(), unquote(parsed.fragment)


def verify(site_root: Path) -> None:
    site_root = site_root.resolve()
    pages = _load_pages(site_root)
    if not pages:
        raise AssertionError(f"no rendered HTML pages found under {site_root}")

    problems: list[str] = []
    for source, parser in pages.items():
        for href in parser.links:
            if href.startswith(SOURCE_PREFIX):
                source_path = unquote(href.removeprefix(SOURCE_PREFIX).split("#", 1)[0])
                if not (PROJECT_ROOT / source_path).is_file():
                    problems.append(
                        f"{source.relative_to(site_root)} -> missing source file {href}"
                    )
                continue
            parsed = urlsplit(href)
            if (
                parsed.scheme
                or parsed.netloc
                or href.startswith(("mailto:", "tel:", "javascript:"))
            ):
                continue
            path_text = unquote(parsed.path)
            if path_text.startswith("/") and not path_text.startswith(SITE_PREFIX):
                problems.append(
                    f"{source.relative_to(site_root)} -> outside the project site {href}"
                )
                continue
            resolved = _local_target(site_root, source, href)
            if resolved is None:
                continue
            target, fragment = resolved
            if not target.is_relative_to(site_root):
                problems.append(
                    f"{source.relative_to(site_root)} -> outside the rendered site {href}"
                )
                continue
            if target.suffix.lower() != ".html":
                if not target.exists():
                    problems.append(f"{source.relative_to(site_root)} -> missing {href}")
                continue
            target_parser = pages.get(target)
            if target_parser is None:
                problems.append(f"{source.relative_to(site_root)} -> missing {href}")
                continue
            if fragment and fragment not in target_parser.anchors:
                problems.append(f"{source.relative_to(site_root)} -> missing anchor {href}")

    if problems:
        details = "\n".join(f"- {problem}" for problem in sorted(set(problems)))
        raise AssertionError(f"broken rendered documentation links:\n{details}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("great-docs/_site")
    verify(root)
