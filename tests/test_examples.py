from __future__ import annotations

import ast
import py_compile
import re
from pathlib import Path

import serpapi_search_tools
from serpapi_search_tools._providers import PROVIDER_SPECS

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

PUBLIC_FACTORIES = {name for name in serpapi_search_tools.__all__ if name.endswith("_search")}
FIXED_ENGINE_FACTORIES = {
    name
    for name, engines in serpapi_search_tools._SUPPORTED_ENGINES_BY_TOOL.items()
    if len(engines) == 1
}


def _example_scripts() -> list[Path]:
    return sorted(path for path in EXAMPLES.glob("*.py") if not path.name.startswith("_"))


def test_examples_start_with_standalone_uv_command() -> None:
    for script in _example_scripts():
        first_line = script.read_text().splitlines()[0]
        assert first_line.startswith("# Run: uv run "), script
        assert f"examples/{script.name}" in first_line, script


def test_examples_cover_every_agent_provider_and_direct_usage() -> None:
    stems = {script.stem for script in _example_scripts()}
    combined = "\n".join(script.read_text() for script in _example_scripts())

    assert {"direct_search", "direct_travel"} <= stems
    for spec in PROVIDER_SPECS:
        if spec.distribution is not None:
            assert f'provider="{spec.name}"' in combined


def test_marketplace_comparison_flattens_real_price_and_link_variants() -> None:
    namespace = {"__name__": "example_test"}
    script = EXAMPLES / "direct_marketplace_comparison.py"
    exec(compile(script.read_text(), str(script), "exec"), namespace)

    normalized = namespace["_first_product"](
        "google_shopping",
        {
            "shopping_results": [
                {
                    "title": "Headphones",
                    "price": {"raw": "$41.99", "extracted": 41.99},
                    "product_link": "https://example.com/headphones",
                }
            ]
        },
    )

    assert normalized == {
        "engine": "google_shopping",
        "title": "Headphones",
        "price": "$41.99",
        "link": "https://example.com/headphones",
    }


def test_agent_logging_client_leaves_compaction_to_the_package_runtime() -> None:
    namespace = {"__name__": "example_test"}
    script = EXAMPLES / "_logging_client.py"
    exec(compile(script.read_text(), str(script), "exec"), namespace)

    response = {
        "search_metadata": {"status": "Success"},
        "search_parameters": {"engine": "google_maps"},
        "local_results": [{"title": "one"}, {"title": "two"}, {"title": "three"}],
        "filters": ["large", "unused"],
    }

    class FakeClient:
        def search(self, params: dict[str, object]) -> dict[str, object]:
            return response

    logging_client = namespace["LoggingClient"].__new__(namespace["LoggingClient"])
    logging_client._client = FakeClient()

    assert logging_client.search({"engine": "google_maps", "q": "coffee"}) == response


def test_examples_are_runnable_scripts_and_compile() -> None:
    for script in _example_scripts():
        source = script.read_text()
        ast.parse(source, filename=str(script))
        assert 'if __name__ == "__main__"' in source
        assert re.search(r"\bsearch_query\s*=", source) is None
        py_compile.compile(str(script), doraise=True)


def test_examples_load_local_env_before_reading_environment() -> None:
    assert not (EXAMPLES / "_env.py").exists()

    for script in _example_scripts():
        source = script.read_text()
        if "os.getenv" in source:
            assert "from dotenv import load_dotenv" in source
            assert "load_dotenv()" in source
            assert source.index("load_dotenv()") < source.index("os.getenv")


def test_examples_cover_every_public_constructor() -> None:
    calls = {name for script in _example_scripts() for name, _ in _factory_calls(script)}

    assert calls == PUBLIC_FACTORIES


def test_example_factory_configuration_respects_vertical_boundaries() -> None:
    for script in _example_scripts():
        for factory, call in _factory_calls(script):
            if factory in FIXED_ENGINE_FACTORIES:
                assert _keyword(call, "allowed_engines") is None
                assert _keyword(call, "default_engine") is None
            if factory == "web_search":
                engines = _literal_strings(_keyword(call, "allowed_engines"))
                assert engines <= {"google", "google_light", "bing", "yahoo", "duckduckgo"}
            if factory == "shopping_search":
                engines = _literal_strings(_keyword(call, "allowed_engines"))
                assert engines <= {"google_shopping", "amazon", "walmart", "ebay"}


def test_google_adk_example_prefers_gemini_key_for_google_api_key() -> None:
    source = (EXAMPLES / "google_adk_gemini.py").read_text()

    assert 'os.getenv("GEMINI_API_KEY")' in source
    assert 'os.environ["GOOGLE_API_KEY"] = gemini_key' in source


def test_google_adk_example_consumes_the_runner_event_stream() -> None:
    script = EXAMPLES / "google_adk_gemini.py"
    tree = ast.parse(script.read_text(), filename=str(script))
    runner_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFor)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and node.iter.func.attr == "run_async"
    ]

    assert len(runner_loops) == 1
    assert not any(isinstance(node, ast.Return) for node in ast.walk(runner_loops[0]))


def test_examples_are_excluded_from_build_artifacts() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    source_exclude = pyproject["tool"]["uv"]["build-backend"]["source-exclude"]
    assert "/examples/**" in source_exclude


def _factory_calls(script: Path) -> list[tuple[str, ast.Call]]:
    tree = ast.parse(script.read_text(), filename=str(script))
    return [
        (node.func.id, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PUBLIC_FACTORIES
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _literal_strings(node: ast.expr | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.List | ast.Tuple):
        return {
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
    return set()
