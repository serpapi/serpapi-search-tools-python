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


def _user_guide_page(filename: str) -> Path:
    matches = tuple(USER_GUIDE.glob(f"[0-9][0-9]-{filename}"))
    assert len(matches) == 1, f"Expected one source page for {filename}, found {matches}"
    return matches[0]


def _workflow_job_text(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)",
        workflow,
    )
    assert match is not None, f"Missing workflow job: {job_name}"
    return match.group(1)


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


def test_great_docs_sources_and_developer_metadata_are_centralized() -> None:
    config = (ROOT / "great-docs.yml").read_text()

    assert USER_GUIDE.is_dir()
    assert SDK_EXAMPLES.is_dir()
    assert not (ROOT / "user_guide").exists()
    assert "user_guide: docs/user_guide" in config
    assert "name: SerpApi" in config
    assert "role: Developer" in config
    assert "email: contact@serpapi.com" in config
    assert "homepage: https://serpapi.com" in config


def test_package_metadata_links_to_public_project_pages() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    urls = pyproject["project"]["urls"]
    assert urls["Documentation"] == "https://serpapi.github.io/serpapi-search-tools-python/"
    assert urls["Repository"] == "https://github.com/serpapi/serpapi-search-tools-python"
    assert urls["Issues"] == "https://github.com/serpapi/serpapi-search-tools-python/issues"


def test_required_docs_source_pages_exist() -> None:
    required_pages = (
        "index.qmd",
        "introduction.qmd",
        "quickstart.qmd",
        "search_tools.qmd",
        "usage.qmd",
        "recipes.qmd",
        "frameworks.qmd",
        "examples.qmd",
        "configuration.qmd",
        "debugging.qmd",
        "testing.qmd",
        "contributing.qmd",
    )

    for filename in required_pages:
        page = _user_guide_page(filename)
        assert page.is_file()
        assert "guide-section:" in page.read_text()


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
        assert f"../sdk-examples/{example_path}.html" in frameworks


def test_sdk_examples_use_auto_detection_in_their_primary_programs() -> None:
    for filename in SUPPORTED_SDK_EXAMPLE_PAGES:
        page = (SDK_EXAMPLES / filename).read_text()
        programs = re.findall(r"```python\s*\n(.*?)```", page, re.S)
        uses_auto_detection = any(
            "_search(" in program and "provider=" not in program for program in programs
        )
        assert uses_auto_detection, filename


def test_docs_assets_are_present_and_configured() -> None:
    config = (ROOT / "great-docs.yml").read_text()
    css_path = ROOT / "assets" / "docs-mobile.css"
    javascript_path = ROOT / "assets" / "docs-mobile.js"
    assert css_path.is_file()
    assert javascript_path.is_file()

    for asset in ("assets/docs-mobile.css", "assets/docs-mobile.js"):
        assert asset in config


def test_docs_document_every_tool_engine_and_closed_input_contract() -> None:
    import serpapi_search_tools

    catalog = _user_guide_page("search_tools.qmd").read_text()
    readme = (ROOT / "README.md").read_text()
    tool_names = {name for name in serpapi_search_tools.__all__ if name.endswith("_search")}
    engines_by_tool = serpapi_search_tools._SUPPORTED_ENGINES_BY_TOOL
    assert set(engines_by_tool) == tool_names
    engines = {engine for values in engines_by_tool.values() for engine in values}

    for tool_name in tool_names:
        assert f"`{tool_name}`" in catalog
    for engine in engines:
        assert f"`{engine}`" in catalog
        assert engine in readme


def test_authored_docs_do_not_use_retired_catch_all_api() -> None:
    paths = [ROOT / "README.md", ROOT / "contributing.md"]
    paths.extend(USER_GUIDE.glob("*.qmd"))
    paths.extend(SDK_EXAMPLES.glob("*.qmd"))
    python = "\n".join(
        source
        for path in paths
        for source in re.findall(r"```python\s*\n(.*?)```", path.read_text(), re.S)
    )

    assert "from serpapi_search_tools import Engine" not in python
    assert "search_query=" not in python
    assert "serpapi_params" not in python


def test_ci_runs_on_branches_and_prs_while_docs_publish_only_on_releases() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    docs = (ROOT / ".github" / "workflows" / "docs.yml").read_text()

    assert "pull_request:" in ci
    assert 'branches: ["**"]' in ci
    assert "contents: read" in ci

    assert re.search(r"(?m)^on:\n  release:\n    types: \[published\]$", docs)
    for trigger in ("pull_request:", "push:", "workflow_dispatch:", "schedule:"):
        assert trigger not in docs
    assert "contents: read" in docs
    assert "uv run --group docs great-docs build" in docs
    assert "python tests/verify_built_docs.py" in docs
    assert "actions/deploy-pages" in docs


def test_testpypi_workflow_is_manual_main_only_and_matches_release_gates() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "testpypi.yml"
    assert workflow_path.is_file()
    workflow = workflow_path.read_text()

    assert re.search(r"(?m)^on:\n  workflow_dispatch:$", workflow)
    for trigger in ("pull_request:", "push:", "release:", "schedule:"):
        assert trigger not in workflow
    assert "contents: read" in workflow
    assert "ref: main" in workflow

    for job in ("test", "build", "verify-distributions", "framework-tests", "publish"):
        assert f"\n  {job}:\n" in workflow

    build = _workflow_job_text(workflow, "build")
    assert "needs: test" in build
    assert "uv build" in build
    assert "testpypi-distributions" in build
    assert "actions/upload-artifact@" in build

    verify = _workflow_job_text(workflow, "verify-distributions")
    assert "needs: build" in verify
    assert "testpypi-distributions" in verify
    assert "tests/clean_install_smoke.py" in verify
    assert "tests/test_live.py" in verify
    assert "SERPAPI_API_KEY: ${{ secrets.SERPAPI_API_KEY }}" in verify
    assert "SERPAPI_KEY: ${{ secrets.SERPAPI_KEY }}" in verify
    assert 'test -n "$SERPAPI_API_KEY" || test -n "$SERPAPI_KEY"' in verify

    frameworks = _workflow_job_text(workflow, "framework-tests")
    assert "needs: test" in frameworks
    assert "tox-environment:" in frameworks

    publish = _workflow_job_text(workflow, "publish")
    assert "needs: [verify-distributions, framework-tests]" in publish
    assert "environment: testpypi" in publish
    assert "id-token: write" in publish
    assert "testpypi-distributions" in publish
    assert "pypa/gh-action-pypi-publish@" in publish
    assert "repository-url: https://test.pypi.org/legacy/" in publish
    assert "PYPI_TOKEN" not in workflow
    assert "password:" not in workflow


def test_ci_separates_quality_basic_framework_and_live_checks() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "pull_request_target" not in ci
    for job in ("quality", "test", "framework-tests", "live-serpapi", "live-adapters"):
        assert f"\n  {job}:\n" in ci

    frameworks = _workflow_job_text(ci, "framework-tests")
    live = _workflow_job_text(ci, "live-serpapi")
    live_adapters = _workflow_job_text(ci, "live-adapters")

    assert "SERPAPI_API_KEY" not in frameworks
    assert "SERPAPI_KEY" not in frameworks
    assert "SERPAPI_API_KEY: ${{ secrets.SERPAPI_API_KEY }}" in live
    assert "SERPAPI_KEY: ${{ secrets.SERPAPI_KEY }}" in live
    assert 'test -n "$SERPAPI_API_KEY" || test -n "$SERPAPI_KEY"' in live
    assert "needs: quality" in live
    assert "github.event_name == 'push' && github.ref == 'refs/heads/main'" in live
    assert "github.event_name == 'schedule'" in live
    assert "github.event_name == 'workflow_dispatch'" in live
    assert "uv run tox -e live" in live
    assert 'python-version: "3.14"' in live

    assert "needs: quality" in live_adapters
    assert "SERPAPI_API_KEY: ${{ secrets.SERPAPI_API_KEY }}" in live_adapters
    assert "SERPAPI_KEY: ${{ secrets.SERPAPI_KEY }}" in live_adapters
    assert 'test -n "$SERPAPI_API_KEY" || test -n "$SERPAPI_KEY"' in live_adapters
    assert "tests/test_live_adapters.py" in live_adapters


def test_release_workflow_uses_pypi_trusted_publishing() -> None:
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    verify_distributions = _workflow_job_text(release, "verify-distributions")

    assert "release:" in release
    assert "types: [published]" in release
    assert "contents: read" in release
    assert 'test "$RELEASE_TAG" = "v$PACKAGE_VERSION"' in release
    assert "uv build" in release
    assert "actions/upload-artifact" in release
    assert "actions/download-artifact" in release
    assert "\n  test:\n" in release
    assert "needs: test" in release
    assert "\n  verify-distributions:\n" in release
    assert "tests/clean_install_smoke.py" in release
    assert "tests/test_live.py" in release
    assert "SERPAPI_API_KEY: ${{ secrets.SERPAPI_API_KEY }}" in verify_distributions
    assert "SERPAPI_KEY: ${{ secrets.SERPAPI_KEY }}" in verify_distributions
    assert 'test -n "$SERPAPI_API_KEY" || test -n "$SERPAPI_KEY"' in verify_distributions
    assert 'python-version: "3.14"' in verify_distributions
    assert "\n  framework-tests:\n" in release
    assert "tox-environment:" in _workflow_job_text(release, "framework-tests")
    assert "needs: [verify-distributions, framework-tests]" in release
    assert "environment: pypi" in release
    assert "id-token: write" in release
    assert "pypa/gh-action-pypi-publish@" in release
    assert "PYPI_TOKEN" not in release
    assert "password:" not in release
