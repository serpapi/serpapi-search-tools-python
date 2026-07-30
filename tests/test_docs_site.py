import ast
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
USER_GUIDE = DOCS / "user_guide"
SDK_EXAMPLES = DOCS / "sdk_examples"
COOKBOOK = DOCS / "cookbook"


def _user_guide_page(filename: str) -> Path:
    matches = tuple(USER_GUIDE.glob(f"[0-9][0-9]-{filename}"))
    assert len(matches) == 1, f"Expected one source page for {filename}, found {matches}"
    return matches[0]


def _supported_sdk_example_pages() -> tuple[str, ...]:
    from serpapi_search_tools._providers import PROVIDER_SPECS

    return tuple(
        f"{spec.name.replace('-', '_')}.qmd"
        for spec in PROVIDER_SPECS
        if spec.distribution is not None
    )


SUPPORTED_SDK_EXAMPLE_PAGES = _supported_sdk_example_pages()

TOOL_DOCS = {
    "web_search": (
        "web_search.qmd",
        {
            "https://serpapi.com/search-api",
            "https://serpapi.com/google-light-api",
            "https://serpapi.com/bing-search-api",
            "https://serpapi.com/yahoo-search-api",
            "https://serpapi.com/duckduckgo-search-api",
        },
    ),
    "news_search": ("news_search.qmd", {"https://serpapi.com/google-news-api"}),
    "maps_search": ("maps_search.qmd", {"https://serpapi.com/google-maps-api"}),
    "images_search": ("images_search.qmd", {"https://serpapi.com/google-images-api"}),
    "shopping_search": (
        "shopping_search.qmd",
        {
            "https://serpapi.com/google-shopping-api",
            "https://serpapi.com/amazon-search-api",
            "https://serpapi.com/walmart-search-api",
            "https://serpapi.com/ebay-search-api",
        },
    ),
    "videos_search": ("videos_search.qmd", {"https://serpapi.com/youtube-search-api"}),
    "hotels_search": ("hotels_search.qmd", {"https://serpapi.com/google-hotels-api"}),
    "flights_search": ("flights_search.qmd", {"https://serpapi.com/google-flights-api"}),
    "travel_explore_search": (
        "travel_explore_search.qmd",
        {"https://serpapi.com/google-travel-explore-api"},
    ),
}


def test_great_docs_dependency_and_config_are_present() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    docs_dependencies = pyproject["dependency-groups"]["docs"]
    assert any(dependency.startswith("great-docs") for dependency in docs_dependencies)

    config = (ROOT / "great-docs.yml").read_text()
    assert "module: serpapi_search_tools" in config
    assert "user_guide: docs/user_guide" in config
    assert "path: src/serpapi_search_tools" in config


def test_great_docs_source_directories_exist() -> None:
    assert USER_GUIDE.is_dir()
    assert SDK_EXAMPLES.is_dir()
    assert COOKBOOK.is_dir()
    assert not (ROOT / "user_guide").exists()


def test_every_public_tool_has_a_complete_dedicated_page_and_navigation_entry() -> None:
    import serpapi_search_tools

    public_tools = {name for name in serpapi_search_tools.__all__ if name.endswith("_search")}
    assert set(TOOL_DOCS) == public_tools

    for tool_name, (filename, engine_urls) in TOOL_DOCS.items():
        path = _user_guide_page(filename)
        assert path.is_file(), path
        page = path.read_text()
        assert f"`{tool_name}`" in page
        for url in engine_urls:
            assert url in page
        assert not re.search(r"https://serpapi\.com/[^)\s]+\.md(?:[)#\s]|$)", page)
        assert "guide-section: Search tools" in page


def test_tool_chooser_links_to_every_dedicated_tool_page() -> None:
    chooser = _user_guide_page("search_tools.qmd").read_text()

    for tool_name, (filename, _) in TOOL_DOCS.items():
        assert f"`{tool_name}`" in chooser
        assert f"({filename.removesuffix('.qmd')}.html)" in chooser


def test_sdk_examples_section_has_one_page_per_supported_sdk() -> None:
    examples_dir = SDK_EXAMPLES
    assert (examples_dir / "index.qmd").is_file()
    for filename in SUPPORTED_SDK_EXAMPLE_PAGES:
        assert (examples_dir / filename).is_file()
    assert not (examples_dir / "plain_callable.qmd").exists()


def test_authored_python_fences_are_valid_python() -> None:
    paths = [ROOT / "README.md", ROOT / "contributing.md"]
    paths.extend(USER_GUIDE.glob("*.qmd"))
    paths.extend(SDK_EXAMPLES.glob("*.qmd"))
    paths.extend(COOKBOOK.glob("*.qmd"))

    for path in paths:
        for index, source in enumerate(
            re.findall(r"```python\s*\n(.*?)```", path.read_text(), re.S)
        ):
            try:
                ast.parse(source)
            except SyntaxError as exc:
                raise AssertionError(f"invalid Python fence {index} in {path}: {exc}") from exc


def test_public_api_has_docstrings() -> None:
    import serpapi_search_tools

    for name in serpapi_search_tools.__all__:
        assert getattr(serpapi_search_tools, name).__doc__, name


def test_every_provider_links_to_its_specific_example() -> None:
    frameworks = _user_guide_page("frameworks.qmd").read_text()

    for filename in SUPPORTED_SDK_EXAMPLE_PAGES:
        example_path = filename.removesuffix(".qmd")
        assert f"../docs/sdk-examples/{example_path}.html" in frameworks


def test_authored_docs_do_not_use_retired_catch_all_api() -> None:
    paths = [ROOT / "README.md", ROOT / "contributing.md"]
    paths.extend(USER_GUIDE.glob("*.qmd"))
    paths.extend(SDK_EXAMPLES.glob("*.qmd"))
    paths.extend(COOKBOOK.glob("*.qmd"))
    python = "\n".join(
        source
        for path in paths
        for source in re.findall(r"```python\s*\n(.*?)```", path.read_text(), re.S)
    )

    assert "from serpapi_search_tools import Engine" not in python
    assert "search_query=" not in python
    assert "serpapi_params" not in python
