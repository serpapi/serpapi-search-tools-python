# Testing

This page describes the contributor test matrix.

## Fast local checks

Run formatting, linting, type checks, the default tests, and a package build:

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest -q -m "not live and not integration and not openai_compat_llm"
uv build
```

The default pytest configuration excludes live tests, framework integration
tests, and the OpenAI-compatible LLM suite.

## Fresh dependency tests with tox

`uv.lock` gives contributors a reproducible development environment. tox checks
something different: it builds the wheel and asks pip to resolve current
packages within the supported ranges in a clean environment.

List the available environments:

```bash
uv run tox list
```

Run one supported Python version:

```bash
uv run tox -e py313
```

Run one framework extra with fresh dependencies:

```bash
uv run tox -e integration-py314-langchain
```

CI runs the base suite on Python 3.10 through 3.14 and installs each framework
extra separately. This catches dependency conflicts that a shared development
environment can hide.

## Newest SDK release canary

The regular integration environments install SDK versions from the supported ranges declared in `pyproject.toml`. The `latest-*` environments install the base wheel without an SDK extra or SDK version constraint. The scheduled workflow then asks PyPI for the newest stable release and installs that exact version, even when it is outside the package's current upper bound. A failure is an early warning to add support; it does not expand the package's declared support range.

GitHub Actions runs this canary every Monday at 05:23 UTC and on manual dispatch. Each matrix job prints its resolved package versions after the test. PyPI pre-releases are excluded unless they become part of pip's normal stable resolution.

Run one canary locally:

```bash
uv run tox -r -e latest-openai-agents
```

## Fake OpenAI-compatible integration tests

These tests start a local FastAPI server. They exercise real framework agent
APIs and tool calls without contacting a model provider or SerpApi:

```bash
uv run --extra frameworks --group openai-compat \
  pytest -q -m integration tests/integration_agents
```

Google ADK uses a separate dependency branch:

```bash
uv run --extra google-adk pytest -q -m integration \
  tests/integration_agents
```

Microsoft Agent Framework also uses an isolated dependency branch:

```bash
uv run --extra microsoft-agent-framework pytest -q -m integration \
  tests/integration_agents
```

The current OpenAI Agents SDK is isolated for the same reason:

```bash
uv run --extra openai-agents pytest -q -m integration \
  tests/integration_agents
```

Missing optional SDKs are skipped. If an installed SDK fails while importing
one of its own dependencies, the test fails instead of hiding the issue as a
skip. Tox sets `SERPAPI_SEARCH_TOOL_REQUIRED_PROVIDER` for each SDK so an
expected provider cannot disappear behind a skip.

When a scenario fails, use
[Debug search responses](../user_guide/20-debugging.qmd) to separate
SerpApi request problems from agent-adapter or model-server problems.

## OpenAI-compatible LLM tests

The opt-in suite can use LM Studio, another local server, or a paid API that
implements OpenAI chat completions. To protect against accidental paid calls,
you must select the marker and set the run flag:

```bash
RUN_OPENAI_COMPAT_LLM_TESTS=1 \
OPENAI_COMPAT_BASE_URL=http://127.0.0.1:1234/v1 \
OPENAI_COMPAT_API_KEY=local-openai-compatible-server \
OPENAI_COMPAT_MODEL=google/gemma-4-e2b \
SERPAPI_API_KEY=... \
uv run --extra frameworks --group openai-compat \
  pytest -q -m openai_compat_llm \
  tests/openai_compat_llm_tests
```

Set `OPENAI_COMPAT_MODEL` for hosted APIs; doing so avoids a model-discovery
request. The suite never falls back to the common `OPENAI_API_KEY` variable.
Together, the shared and isolated environments run every SDK that exposes an
OpenAI-compatible agent path with a fake SerpApi client, exercise the top SDKs
across representative input shapes, and keep one separate real-SerpApi
end-to-end case. Microsoft Agent Framework and OpenAI Agents use isolated
dependency branches. Claude Agent SDK and Semantic Kernel are covered by
deterministic native-handler invocation tests because their execution APIs do
not use this common chat-completions harness.

The same suites are available as isolated tox environments:

```bash
uv run tox -e openai-compat
uv run tox -e openai-compat-google-adk
uv run tox -e openai-compat-microsoft-agent-framework
uv run tox -e openai-compat-openai-agents
```

## Live SerpApi tests

```bash
SERPAPI_API_KEY=... uv run pytest -q -m live
```

These tests spend SerpApi quota. They cover every advertised engine, including
Yahoo, Amazon, Walmart, eBay, and YouTube's native query keys and structured
hotels, flights, and travel-explore requests. The core suite runs on Python 3.14
and checks non-empty result families, typed parameter effects, Unicode, error
contracts, cache modes, and parameter combinations copied from the docs.

CI also makes one real Google Light request through each of the 14 optional SDK
adapters. These adapter tests invoke the native tool interface directly and do
not require an LLM key. CrewAI and LangGraph currently use Python 3.13; the
other adapter lanes use Python 3.14.

The live jobs run on trusted pushes to `main`, manual dispatches, and the
existing Monday schedule. The scheduled run additionally probes upstream
parameter combinations that the package rejects locally and uploads a JSON
inventory of the top-level result keys. A complete weekly run makes roughly
47 SerpApi requests before provider-side cache reuse. Keep keys in the
environment and do not enable live jobs for untrusted fork pull requests.

## Build the docs

Great Docs 0.14 requires Python 3.11 or newer:

```bash
uv run --group docs great-docs build
uv run python scripts/fix_built_docs_links.py
uv run python tests/verify_built_docs.py
```

The generated `great-docs/` directory is temporary and ignored by git.
