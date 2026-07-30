# OpenAI Agents managed research report

A report editor delegates focused investigations to a search specialist and
retains ownership of the final answer. The result is written to
`cookbook-output/openai-agents-research-report.md`.

## Original source

Inspired by the official OpenAI Agents SDK
[research bot example](https://github.com/openai/openai-agents-python/tree/main/examples/research_bot),
which separates research work from report synthesis.

This adaptation was **enhanced with SerpApi** by equipping the research
specialist with typed web and news tools and requiring explicit source URLs and
uncertainty in its handoff to the editor.

## Run

Set `SERPAPI_API_KEY` and `OPENAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --with '.[openai-agents]' cookbook/openai-agents/main.py
```

Optional environment variables: `OPENAI_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.
