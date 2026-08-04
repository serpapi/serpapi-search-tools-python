# CrewAI collaborative trip planner

A search researcher gathers live flight, hotel, and map evidence; a second
agent turns that packet into a budget-aware trip decision. The plan is written
to `cookbook-output/crewai-trip-plan.md`.

## Original source

Inspired by the official CrewAI
[Trip Planner example](https://github.com/crewAIInc/crewAI-examples), which
demonstrates role-based research and planning with a sequential crew.

This adaptation was **enhanced with SerpApi** by replacing generic search with
typed Google Flights, Google Hotels, and Google Maps tools. Required dates,
airport identifiers, and occupancy are visible to the model through their tool
schemas.

## Run

Set `SERPAPI_API_KEY` and `XAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --isolated --no-project --with 'serpapi-search-tools[crewai]' --with python-dotenv cookbook/crewai/main.py
```

Optional environment variables: `XAI_MODEL`, `XAI_BASE_URL`,
`COOKBOOK_PROMPT`, and `COOKBOOK_OUTPUT_DIR`.

