# Agno market research report

This Agno agent researches category structure, product launches, and live
marketplace signals before writing a report to
`cookbook-output/agno-market-research.md`.

## Original source

Inspired by Agno's official
[parallel market research cookbook](https://github.com/agno-agi/agno/blob/main/cookbook/91_tools/parallel/market_research.py).

This adaptation was **enhanced with SerpApi** by providing web, news, and
shopping tools with distinct result surfaces, replacing the original source's
search dependency while retaining its decision-focused market-research goal.

## Run

Set `SERPAPI_API_KEY` and `XAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --with '.[agno]' cookbook/agno/main.py
```

Optional environment variables: `XAI_MODEL`, `XAI_BASE_URL`,
`COOKBOOK_PROMPT`, and `COOKBOOK_OUTPUT_DIR`.
