# LangChain deep market research

This agent produces a source-backed market brief instead of a one-shot search
answer. It plans the investigation, separates recent reporting from durable
background material, and writes the final memo to
`cookbook-output/langchain-market-research.md`.

## Original source

Inspired by LangChain's official
[Deep Agents from scratch](https://docs.langchain.com/oss/python/langchain/deep-agent-from-scratch)
guide and its
[companion GitHub repository](https://github.com/langchain-ai/deep-agents-from-scratch).

This adaptation was **enhanced with SerpApi** by replacing the original search
dependency with typed `web_search` and `news_search` tools from
`serpapi-search-tools`.

## Run

Set `SERPAPI_API_KEY` and `XAI_API_KEY` in the repository-root `.env`, then run:

```bash
uv run --with . --with langchain --with langchain-openai \
  cookbook/langchain/main.py
```

Optional environment variables: `XAI_MODEL`, `XAI_BASE_URL`,
`COOKBOOK_PROMPT`, and `COOKBOOK_OUTPUT_DIR`.

