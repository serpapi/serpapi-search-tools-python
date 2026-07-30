# LlamaIndex remote-work destination brief

This FunctionAgent combines destination exploration, web research, and local
place search to recommend a short remote-work trip. It writes the result to
`cookbook-output/llamaindex-destination-brief.md`.

## Original source

Inspired by LlamaIndex's official
[basic agent workflow](https://github.com/run-llama/llama_index/blob/main/docs/examples/agent/agent_workflow_basic.ipynb)
and its tool-driven `FunctionAgent` pattern.

This adaptation was **enhanced with SerpApi** by adding typed travel-explore,
web, and maps tools rather than relying on a single generic function.

## Run

Set `SERPAPI_API_KEY` and `OPENAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --with '.[llamaindex]' --with llama-index-llms-openai \
  cookbook/llamaindex/main.py
```

Optional environment variables: `OPENAI_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.
