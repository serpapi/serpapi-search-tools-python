# Semantic Kernel plan-and-execute competitor brief

This ChatCompletionAgent follows an explicit plan, resolves the plan with
search plugins, and writes a competitor brief to
`cookbook-output/semantic-kernel-competitor-brief.md`.

## Original source

Inspired by Semantic Kernel's official Python
[plan-and-execute process](https://github.com/microsoft/semantic-kernel/blob/main/python/samples/concepts/processes/plan_and_execute.py).

This adaptation was **enhanced with SerpApi** by replacing the sample's
provider-native web search with typed web and news plugins and by applying the
workflow to a concrete competitor-analysis artifact.

## Run

Set `SERPAPI_API_KEY` and `OPENAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --isolated --no-project --with 'serpapi-search-tools[semantic-kernel]' --with python-dotenv \
  cookbook/semantic-kernel/main.py
```

Optional environment variables: `OPENAI_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.
