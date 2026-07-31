# Pydantic AI visual location scout

This agent researches campaign locations using visual references, place data,
and practical web verification. It writes the final scouting brief to
`cookbook-output/pydantic-ai-location-scout.md`.

## Original source

Inspired by Pydantic AI's official
[weather agent example](https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/weather-agent.md),
which demonstrates a focused agent using typed external tools.

This adaptation was **enhanced with SerpApi** by replacing the demonstration
weather/geocoding functions with typed image, maps, and web search tools and by
requiring cross-verification before recommending a location.

## Run

Set `SERPAPI_API_KEY` and `OPENAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --isolated --no-project --with 'serpapi-search-tools[pydantic-ai]' --with python-dotenv cookbook/pydantic-ai/main.py
```

Optional environment variables: `OPENAI_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.

