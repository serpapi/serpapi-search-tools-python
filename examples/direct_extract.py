# Run: uv run --isolated --no-project --with . --with python-dotenv examples/direct_extract.py map https://www.python.org/about/
"""Read a page, discover site URLs, or crawl up to two pages as Markdown."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from serpapi_search_tools import web_crawl, web_extract, web_map

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("scrape", "map", "crawl"), nargs="?", default="scrape"
    )
    parser.add_argument("url", nargs="?", default="https://example.com")
    args = parser.parse_args()
    if args.operation == "crawl":
        tool = web_crawl(
            provider="function", timeout=90, default_params={"max_pages": 2, "max_depth": 1}
        )
    elif args.operation == "map":
        tool = web_map(provider="function", timeout=90)
    else:
        tool = web_extract(provider="function", timeout=90)
    print(tool(url=args.url))


if __name__ == "__main__":
    main()
