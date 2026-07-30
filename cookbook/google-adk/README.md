# Google ADK retail location strategy

This ADK agent compares candidate cities with place-level, durable, and current
evidence before writing a location strategy memo to
`cookbook-output/google-adk-location-strategy.md`.

## Original source

Inspired by Google's official ADK
[retail AI location strategy sample](https://github.com/google/adk-samples/tree/main/python/agents/retail-ai-location-strategy).

This adaptation was **enhanced with SerpApi** by replacing the sample's search
stack with typed maps, web, and news tools that can be used together through
the `serpapi-search-tools` Google ADK adapter.

## Run

Set `SERPAPI_API_KEY` and `GEMINI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --with . --with google-adk cookbook/google-adk/main.py
```

Optional environment variables: `GOOGLE_ADK_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.

