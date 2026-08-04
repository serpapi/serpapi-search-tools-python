# AutoGen company intelligence

This AssistantAgent iteratively gathers company facts and current reporting,
then reflects on its tool results before writing an intelligence memo to
`cookbook-output/autogen-company-intelligence.md`.

## Original source

Inspired by AutoGen's official
[company research example](https://github.com/microsoft/autogen/blob/main/python/docs/src/user-guide/agentchat-user-guide/examples/company-research.ipynb).

This adaptation was **enhanced with SerpApi** by replacing the example's search
surface with separate typed web and news tools and by requiring the agent to
distinguish company claims from independent reporting.

## Run

Set `SERPAPI_API_KEY` and `OPENAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --isolated --no-project --with 'serpapi-search-tools[autogen]' --with python-dotenv cookbook/autogen/main.py
```

Optional environment variables: `OPENAI_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.
