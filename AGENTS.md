# AGENTS.md

## Project Structure

| Path | Purpose |
| --- | --- |
| `src/serpapi_search_tools/_shared.py` | Provider-neutral definitions, client runtime, schemas, and validation helpers |
| `src/serpapi_search_tools/_query_tools.py` | Web, news, maps, images, shopping, and video constructors |
| `src/serpapi_search_tools/_travel_tools.py` | Hotels, flights, and travel-explore constructors |
| `src/serpapi_search_tools/_providers.py` | Agent SDK registry, aliases, extras, and auto-detection |
| `src/serpapi_search_tools/_adapters.py` | Lazy native agent SDK adapters |
| `tests/` | Default unit, metadata, example, live, and fake OpenAI-compatible tests |
| `tests/integration_agents/` | FastAPI fake OpenAI-compatible server and agent framework scenarios |
| `tests/openai_compat_llm_tests/` | Opt-in tests for local or hosted OpenAI-compatible APIs, excluded by default |
| `examples/` | Runnable local examples, excluded from package artifacts |
| `docs/user_guide/` | Great Docs user-guide source pages |
| `docs/sdk_examples/` | Great Docs SDK example source pages rendered at `/docs/sdk-examples/` |
| `great-docs.yml` | Great Docs config and API reference layout |

## Testing

Use targeted tests while changing code, then run the default checks:

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest -q -m "not live and not integration and not openai_compat_llm"
uv build
```

Fresh compatibility environments resolve supported package ranges without
using `uv.lock`:

```bash
uv run tox -e py313
uv run tox -e integration-py314-langchain
```

Adapter integration tests use the fake OpenAI-compatible FastAPI server:

```bash
uv run --extra frameworks --group openai-compat \
  pytest -q -m integration tests/integration_agents
uv run --extra google-adk pytest -q -m integration tests/integration_agents
uv run --extra openai-agents pytest -q -m integration tests/integration_agents
```

Live SerpApi tests require a key:

```bash
SERPAPI_API_KEY=... uv run tox -e live
```

OpenAI-compatible LLM tests are opt-in and require both a marker and run flag:

```bash
RUN_OPENAI_COMPAT_LLM_TESTS=1 \
OPENAI_COMPAT_BASE_URL=http://127.0.0.1:1234/v1 \
OPENAI_COMPAT_API_KEY=local-openai-compatible-server \
OPENAI_COMPAT_MODEL=google/gemma-4-e2b \
SERPAPI_API_KEY=... \
uv run tox -e openai-compat
```

Run the Google ADK OpenAI-compatible smoke separately:

```bash
RUN_OPENAI_COMPAT_LLM_TESTS=1 \
OPENAI_COMPAT_BASE_URL=http://127.0.0.1:1234/v1 \
OPENAI_COMPAT_API_KEY=local-openai-compatible-server \
OPENAI_COMPAT_MODEL=google/gemma-4-e2b \
uv run tox -e openai-compat-google-adk
```

Run the Microsoft Agent Framework OpenAI-compatible smoke in its isolated
dependency branch:

```bash
RUN_OPENAI_COMPAT_LLM_TESTS=1 \
OPENAI_COMPAT_BASE_URL=http://127.0.0.1:1234/v1 \
OPENAI_COMPAT_API_KEY=local-openai-compatible-server \
OPENAI_COMPAT_MODEL=google/gemma-4-e2b \
uv run tox -e openai-compat-microsoft-agent-framework
```

Run the current OpenAI Agents SDK smoke in its isolated dependency branch:

```bash
RUN_OPENAI_COMPAT_LLM_TESTS=1 \
OPENAI_COMPAT_BASE_URL=http://127.0.0.1:1234/v1 \
OPENAI_COMPAT_API_KEY=local-openai-compatible-server \
OPENAI_COMPAT_MODEL=google/gemma-4-e2b \
uv run tox -e openai-compat-openai-agents
```

Docs use Great Docs:

```bash
uv run --group docs great-docs build
uv run python scripts/fix_built_docs_links.py
uv run python tests/verify_built_docs.py
```

Great Docs `0.14` requires Python `3.11+`. The generated `great-docs/`
directory is ephemeral and ignored by git.

## CI Notes

CI runs on pull requests and pushes to `main`. Pull request workflows use
`pull_request`, not `pull_request_target`; maintainer approval for
outside-collaborator PR runs is a GitHub repository setting. Public fork pull
requests never receive repository secrets. Framework jobs use the fake
OpenAI-compatible server; live SerpApi jobs are limited to trusted main pushes,
the weekly schedule, and manual runs. Documentation builds and deploys only
when a GitHub Release is published.

## Releases

`.github/workflows/release.yml` publishes GitHub Releases through PyPI Trusted
Publishing. A tag such as `v0.1.0` must match `project.version`, and the PyPI
publisher must be configured for repository `serpapi-search-tools-python`,
workflow `release.yml`, and GitHub environment `pypi`. No stored PyPI token is
used.

`.github/workflows/testpypi.yml` is a manual main-only rehearsal with the same
quality, exact-wheel live, and adapter gates. Configure its TestPyPI Trusted
Publisher for workflow `testpypi.yml` and GitHub environment `testpypi`.
Duplicate versions fail intentionally; bump `project.version` before another
TestPyPI upload.
