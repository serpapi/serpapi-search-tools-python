import importlib
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_and_import_package_use_plural_search_tools_name() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["project"]["name"] == "serpapi-search-tools"
    assert (ROOT / "src" / "serpapi_search_tools").is_dir()


def test_public_package_exposes_intent_specific_tools_and_semantic_enums() -> None:
    package = importlib.import_module("serpapi_search_tools")

    assert set(package.__all__) == {
        "ShoppingSearchEngine",
        "SearchResultFormat",
        "SearchResultMode",
        "SerpApiSearchError",
        "TravelClass",
        "WebSearchEngine",
        "flights_search",
        "hotels_search",
        "images_search",
        "maps_search",
        "news_search",
        "shopping_search",
        "travel_explore_search",
        "videos_search",
        "web_search",
    }
    assert all(callable(getattr(package, name)) for name in package.__all__)


def test_base_install_declares_runtime_dependencies() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime_dependencies = pyproject["project"]["dependencies"]
    dev_dependencies = pyproject["dependency-groups"]["dev"]

    assert pyproject["project"]["requires-python"] == ">=3.10"
    assert any(dependency.startswith("serpapi") for dependency in runtime_dependencies)
    assert any(dependency.startswith("mistune") for dependency in runtime_dependencies)
    assert all("dotenv" not in dependency for dependency in runtime_dependencies)
    assert any(dependency.startswith("python-dotenv") for dependency in dev_dependencies)


def test_frameworks_are_optional_extras() -> None:
    from serpapi_search_tools._providers import PROVIDER_SPECS

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    optional = pyproject["project"]["optional-dependencies"]
    conflicts = pyproject["tool"]["uv"]["conflicts"]

    expected_extras = {spec.extra for spec in PROVIDER_SPECS if spec.extra is not None}
    assert set(optional) == expected_extras | {"frameworks"}
    assert all("serpapi" not in dependency for deps in optional.values() for dependency in deps)
    assert _has_uv_conflict(conflicts, "google-adk", "langchain")
    assert _has_uv_conflict(conflicts, "google-adk", "langgraph")
    assert not _has_uv_conflict(conflicts, "frameworks", "microsoft-agent-framework")
    assert _has_uv_conflict(conflicts, "frameworks", "openai-agents")
    assert _has_uv_conflict(conflicts, "openai-agents", "semantic-kernel")


def test_declared_major_version_bounds_match_tested_framework_branches() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    optional = pyproject["project"]["optional-dependencies"]
    openai_compat = pyproject["dependency-groups"]["openai-compat"]

    assert "openai>=1.0.0,<3" in optional["frameworks"]
    assert "pydantic-ai>=1.30.1,<3" in optional["frameworks"]
    assert optional["pydantic-ai"] == ["pydantic-ai>=1.30.1,<3"]
    assert "haystack-ai>=2.30.2,<4" in optional["frameworks"]
    assert optional["haystack"] == ["haystack-ai>=2.30.2,<4"]
    assert optional["llamaindex"] == ["llama-index-core>=0.14.22,<0.15"]
    assert openai_compat == [
        "langchain-openai>=1.1.9,<2",
        "llama-index-llms-openai>=0.6.26,<0.8",
    ]
    assert all(
        not dependency.startswith(("langchain-openai", "llama-index-llms-openai"))
        for dependencies in optional.values()
        for dependency in dependencies
    )
    assert all(
        re.match(r"^pydantic(?:\s|[<>=!~;]|$)", dependency) is None
        for dependencies in optional.values()
        for dependency in dependencies
    )
    assert all(
        not classifier.startswith("License ::")
        for classifier in pyproject["project"]["classifiers"]
    )


def test_tox_has_a_fresh_environment_for_every_framework_extra() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    tox_config = (ROOT / "tox.ini").read_text()

    for extra in pyproject["project"]["optional-dependencies"]:
        if extra == "frameworks":
            continue
        factorized_extra = f"{extra}: {extra}"
        standalone_extra = f"extras = {extra}"
        assert factorized_extra in tox_config or standalone_extra in tox_config


def test_github_actions_are_pinned_to_commit_shas() -> None:
    uses_lines = [
        line.strip()
        for workflow in (ROOT / ".github" / "workflows").glob("*.yml")
        for line in workflow.read_text().splitlines()
        if "uses:" in line
    ]

    assert uses_lines
    for line in uses_lines:
        reference = line.split("uses:", maxsplit=1)[1].strip().split()[0]
        assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference), line


def test_openai_compat_llm_tests_are_documented_and_excluded_by_default() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]

    assert pytest_options["testpaths"] == ["tests"]
    assert "not openai_compat_llm" in pytest_options["addopts"]
    assert any(marker.startswith("openai_compat_llm:") for marker in pytest_options["markers"])
    assert (ROOT / "tests" / "openai_compat_llm_tests").is_dir()


def test_workflows_avoid_unsafe_triggers_and_static_publish_credentials() -> None:
    workflows = {
        path.name: path.read_text() for path in (ROOT / ".github" / "workflows").glob("*.yml")
    }

    assert workflows
    assert all("pull_request_target" not in workflow for workflow in workflows.values())
    for filename in ("release.yml", "testpypi.yml"):
        workflow = workflows[filename]
        assert "id-token: write" in workflow
        assert "PYPI_TOKEN" not in workflow
        assert "password:" not in workflow


def _has_uv_conflict(conflicts: list[list[dict[str, str]]], *extras: str) -> bool:
    expected = {extra for extra in extras}
    return any(
        {item.get("extra") for item in conflict if "extra" in item} >= expected
        for conflict in conflicts
    )
