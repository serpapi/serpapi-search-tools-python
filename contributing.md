# Contributing

Thanks for improving `serpapi-search-tools`. The public API is organized around
capability-specific constructors and a separate agent-SDK adapter layer.

## Architecture

| Path | Purpose |
| --- | --- |
| `src/serpapi_search_tools/_shared.py` | Tool definitions, client runtime, schema helpers, and shared validation |
| `src/serpapi_search_tools/_query_tools.py` | Web, news, maps, images, shopping, and video tools |
| `src/serpapi_search_tools/_travel_tools.py` | Hotels, flights, and travel exploration |
| `src/serpapi_search_tools/_providers.py` | Provider registry, aliases, extras, and detection |
| `src/serpapi_search_tools/_adapters.py` | Lazy native SDK adapters |
| `tests/` | Unit, schema, integration, live, docs, and packaging tests |
| `examples/` | Direct and SDK-specific runnable examples |

## Adding a new search tool

Start from the engine's official page in
[SerpApi's LLM documentation index](https://serpapi.com/llms.txt). Give the new
constructor an explicit typed signature and a closed JSON Schema. Translate
semantic application values to native SerpApi parameter names internally, and
validate required and incompatible inputs before making a paid request.

Add unit tests for exact request construction, adapter-matrix schema tests, a
live case, documentation, and an example. Advanced model-controlled inputs
should become typed fields; never expose an unrestricted parameters dictionary
to an agent.

## Adding a new SDK adapter

Register provider metadata in `_providers.py`, implement one lazy conversion in
`_adapters.py`, and add its optional dependency and tox environment. The
adapter consumes the same provider-neutral `ToolDefinition` for all search
constructors, so it must not contain engine-specific logic.

Test native construction, the exact schema, missing-dependency errors, and an
actual tool call against the fake model server where supported. Add one
runnable example and one page under `docs/sdk_examples/`.

## Default checks

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest -q -m "not live and not integration and not openai_compat_llm"
uv build
uv run --group docs great-docs build
uv run python scripts/fix_built_docs_links.py
uv run python tests/verify_built_docs.py
```

See `docs/user_guide/30-testing.qmd` for the complete tox, integration, live SerpApi,
and OpenAI-compatible model matrix.

OpenAI-compatible model tests are opt-in and require
`RUN_OPENAI_COMPAT_LLM_TESTS=1` plus `OPENAI_COMPAT_BASE_URL`,
`OPENAI_COMPAT_API_KEY`, and `OPENAI_COMPAT_MODEL`. This prevents accidental
model calls during normal development. The suite lives in
`tests/openai_compat_llm_tests`; `google/gemma-4-e2b` is the preferred local
LM Studio model when it is installed.

Deterministic SDK integration scenarios and schema matrices live in
`tests/integration_agents` and use a local fake OpenAI-compatible server.

## Publish a main-branch build to TestPyPI

The **Publish main to TestPyPI** GitHub Actions workflow is manual. Every source
job explicitly checks out `main`, then runs the same source checks, exact-wheel
smoke and live tests, and 14-adapter matrix used by the production PyPI release.

Before the first run:

1. Create a protected GitHub environment named `testpypi` and restrict its
   deployment branches to `main`.
2. Configure a TestPyPI Trusted Publisher for repository
   `serpapi-search-tools-python`, workflow `testpypi.yml`, and environment
   `testpypi`.
3. Keep `SERPAPI_API_KEY` available as a repository secret for the exact-wheel
   live verification job.

No TestPyPI token is stored in GitHub. The publish job requests a short-lived
OIDC credential only after every verification job passes.

Run the workflow from the GitHub Actions page and select `main`. TestPyPI, like
PyPI, does not allow the same project version to be uploaded twice. The
workflow deliberately fails for a duplicate version instead of skipping the
upload or changing package metadata. Bump `project.version` before publishing a
new TestPyPI build.

## Publish documentation

`.github/workflows/docs.yml` builds and deploys documentation only when a
GitHub Release is published. It renders Great Docs from the release source,
repairs and verifies generated links, and deploys the verified artifact to
GitHub Pages. Pull requests and ordinary branch pushes rely on the local docs
commands above rather than running a separate docs workflow.

Do not commit generated `great-docs/` output or credentials.
