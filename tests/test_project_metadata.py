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


def test_base_install_only_depends_on_serpapi() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime_dependencies = pyproject["project"]["dependencies"]
    dev_dependencies = pyproject["dependency-groups"]["dev"]

    assert pyproject["project"]["requires-python"] == ">=3.10"
    assert any(dependency.startswith("serpapi") for dependency in runtime_dependencies)
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


def test_langgraph_extra_installs_langgraph_and_langchain_tool_dependencies() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["optional-dependencies"]["langgraph"]

    assert any(dependency.startswith("langgraph>=1.2.6,<2") for dependency in dependencies)
    assert any(dependency.startswith("langchain-core") for dependency in dependencies)
    assert any(dependency.startswith("langchain-openai") for dependency in dependencies)


def test_dev_tooling_includes_ruff_and_ty() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev_dependencies = pyproject["dependency-groups"]["dev"]

    assert any(dependency.startswith("ruff") for dependency in dev_dependencies)
    assert any(dependency.startswith("ty") for dependency in dev_dependencies)


def test_tox_tests_supported_python_versions_with_fresh_environments() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev_dependencies = pyproject["dependency-groups"]["dev"]
    tox_config = (ROOT / "tox.ini").read_text()

    assert any(dependency == "tox>=4.22,<5" for dependency in dev_dependencies)
    for environment in ("py310", "py311", "py312", "py313", "py314"):
        assert environment in tox_config.split("[testenv]", maxsplit=1)[0]
    assert "package = wheel" in tox_config
    assert "[testenv:clean-install]" in tox_config
    assert "commands = python tests/clean_install_smoke.py" in tox_config
    assert "[testenv:live]" in tox_config
    assert "commands = python -m pytest -q -m live tests/test_live.py" in tox_config
    assert "[testenv:openai-compat]" in tox_config
    assert "[testenv:openai-compat-google-adk]" in tox_config
    for environment_name in (
        "OPENAI_COMPAT_API_KEY",
        "OPENAI_COMPAT_BASE_URL",
        "OPENAI_COMPAT_MODEL",
        "RUN_OPENAI_COMPAT_LLM_TESTS",
    ):
        assert environment_name in tox_config
    assert "uv.lock" not in tox_config


def test_tox_has_a_fresh_environment_for_every_framework_extra() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    tox_config = (ROOT / "tox.ini").read_text()

    for extra in pyproject["project"]["optional-dependencies"]:
        if extra == "frameworks":
            continue
        factorized_extra = f"{extra}: {extra}"
        standalone_extra = f"extras = {extra}"
        assert factorized_extra in tox_config or standalone_extra in tox_config


def test_ci_routes_test_matrices_through_tox() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "uv run tox -e ${{ matrix.tox-environment }}" in workflow
    assert "uv sync --dev --extra ${{ matrix.extra }}" not in workflow


def test_google_adk_python_310_environment_includes_litellm_backend() -> None:
    tox_config = (ROOT / "tox.ini").read_text()

    assert "[testenv:integration-py{310,314}-google-adk]" in tox_config
    section = tox_config.split("[testenv:integration-py{310,314}-google-adk]", maxsplit=1)[1].split(
        "[testenv:", maxsplit=1
    )[0]
    assert "py310: litellm" in section
    assert "py310: orjson" in section


def test_gitignore_covers_python_artifacts_and_local_env_files() -> None:
    gitignore = (ROOT / ".gitignore").read_text().splitlines()

    for pattern in (
        "__pycache__/",
        ".tox/",
        ".venv/",
        "dist/",
        "great-docs/",
        ".env",
        "*.env",
        "!.env.example",
        "!examples/sample.env",
    ):
        assert pattern in gitignore


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


def test_uv_build_backend_is_configured() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["build-system"]["build-backend"] == "uv_build"
    assert any(
        dependency.startswith("uv_build") for dependency in pyproject["build-system"]["requires"]
    )
    assert "tool" not in pyproject or "hatch" not in pyproject.get("tool", {})
    source_exclude = set(pyproject["tool"]["uv"]["build-backend"]["source-exclude"])
    assert {"/examples/**", "/great-docs/**"} <= source_exclude


def test_openai_compat_llm_tests_are_documented_and_excluded_by_default() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]

    assert pytest_options["testpaths"] == ["tests"]
    assert "not openai_compat_llm" in pytest_options["addopts"]
    assert any(marker.startswith("openai_compat_llm:") for marker in pytest_options["markers"])
    assert (ROOT / "tests" / "openai_compat_llm_tests").is_dir()


def _has_uv_conflict(conflicts: list[list[dict[str, str]]], *extras: str) -> bool:
    expected = {extra for extra in extras}
    return any(
        {item.get("extra") for item in conflict if "extra" in item} >= expected
        for conflict in conflicts
    )
