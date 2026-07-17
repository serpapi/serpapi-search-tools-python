from pathlib import Path

import pytest


def _write_module(path: Path, source: str = "") -> None:
    path.write_text(source, encoding="utf-8")


def test_import_optional_returns_installed_module(tmp_path, monkeypatch):
    _write_module(tmp_path / "available_optional.py", "VALUE = 42\n")
    monkeypatch.syspath_prepend(tmp_path)

    from optional_dependencies import import_optional

    module = import_optional("available_optional")

    assert module.VALUE == 42


def test_import_optional_skips_missing_requested_module():
    from optional_dependencies import import_optional

    with pytest.raises(pytest.skip.Exception, match="optional dependency"):
        import_optional("definitely_missing_optional_package")


def test_import_optional_skips_missing_requested_submodule(tmp_path, monkeypatch):
    package = tmp_path / "available_parent"
    package.mkdir()
    _write_module(package / "__init__.py")
    monkeypatch.syspath_prepend(tmp_path)

    from optional_dependencies import import_optional

    with pytest.raises(pytest.skip.Exception, match="optional dependency"):
        import_optional("available_parent.missing_child")


def test_import_optional_fails_for_missing_required_provider_module(monkeypatch):
    monkeypatch.setenv("SERPAPI_SEARCH_TOOL_REQUIRED_PROVIDER", "pydantic-ai")

    import optional_dependencies

    def missing_module(module_name: str):
        error = ModuleNotFoundError(f"No module named {module_name!r}")
        error.name = module_name
        raise error

    monkeypatch.setattr(optional_dependencies, "import_module", missing_module)

    with pytest.raises(BaseException) as exc_info:
        optional_dependencies.import_optional("pydantic_ai.definitely_missing_agent_api")

    assert isinstance(exc_info.value, pytest.fail.Exception)
    assert "required provider" in str(exc_info.value)


def test_import_optional_reraises_missing_transitive_module(tmp_path, monkeypatch):
    package = tmp_path / "broken_optional"
    package.mkdir()
    _write_module(
        package / "__init__.py",
        "import missing_transitive_dependency\n",
    )
    monkeypatch.syspath_prepend(tmp_path)

    from optional_dependencies import import_optional

    with pytest.raises(ModuleNotFoundError) as exc_info:
        import_optional("broken_optional")

    assert exc_info.value.name == "missing_transitive_dependency"
