from __future__ import annotations

import ast
import py_compile
from pathlib import Path

import serpapi_search_tools
from serpapi_search_tools._providers import PROVIDER_SPECS

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
COOKBOOK = ROOT / "cookbook"
COOKBOOK_DOCS = ROOT / "docs" / "cookbook"
SUPPORTED_PROVIDERS = {spec.name for spec in PROVIDER_SPECS if spec.distribution is not None}
PUBLIC_FACTORIES = {name for name in serpapi_search_tools.__all__ if name.endswith("_search")}


def _entry_directories() -> list[Path]:
    return sorted(path for path in COOKBOOK.iterdir() if path.is_dir())


def test_cookbook_has_one_entry_for_every_supported_provider() -> None:
    entries = {path.name for path in _entry_directories()}

    assert entries == SUPPORTED_PROVIDERS
    for entry in _entry_directories():
        assert (entry / "main.py").is_file()
        assert (entry / "README.md").is_file()


def test_cookbook_scripts_are_standalone_compile_and_use_the_matching_adapter() -> None:
    for entry in _entry_directories():
        script = entry / "main.py"
        source = script.read_text()
        first_line = source.splitlines()[0]

        assert first_line.startswith("# Run: uv run "), script
        assert f"cookbook/{entry.name}/main.py" in first_line, script
        assert f'provider="{entry.name}"' in source, script
        assert 'if __name__ == "__main__"' in source, script
        assert "load_dotenv()" in source, script
        assert "COOKBOOK_PROMPT" in source, script
        assert "COOKBOOK_OUTPUT_DIR" in source, script
        ast.parse(source, filename=str(script))
        py_compile.compile(str(script), doraise=True)


def test_cookbook_covers_every_public_search_constructor() -> None:
    calls: set[str] = set()
    for entry in _entry_directories():
        tree = ast.parse((entry / "main.py").read_text())
        calls.update(
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in PUBLIC_FACTORIES
        )

    assert calls == PUBLIC_FACTORIES


def test_cookbook_environment_template_matches_the_scripts() -> None:
    sample = (COOKBOOK / "sample.env").read_text()

    for name in (
        "SERPAPI_API_KEY",
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "COOKBOOK_PROMPT",
        "COOKBOOK_OUTPUT_DIR",
    ):
        assert f"{name}=" in sample


def test_cookbook_docs_cover_every_supported_provider() -> None:
    assert (COOKBOOK_DOCS / "index.qmd").is_file()
    pages = {path.stem for path in COOKBOOK_DOCS.glob("*.qmd") if path.name != "index.qmd"}
    assert pages == SUPPORTED_PROVIDERS

    index = (COOKBOOK_DOCS / "index.qmd").read_text()
    for provider in SUPPORTED_PROVIDERS:
        page = (COOKBOOK_DOCS / f"{provider}.qmd").read_text()
        assert f"({provider}.html)" in index
        assert "SerpApi enhancement:" in page
        assert f"cookbook/{provider}/main.py" in page


def test_cookbook_is_documented_but_excluded_from_distributions() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    source_exclude = pyproject["tool"]["uv"]["build-backend"]["source-exclude"]
    config = (ROOT / "great-docs.yml").read_text()

    assert "/cookbook/**" in source_exclude
    assert "title: Cookbook" in config
    assert "dir: docs/cookbook" in config
