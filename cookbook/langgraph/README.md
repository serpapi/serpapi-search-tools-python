# LangGraph product-launch intelligence

This stateful research graph loops between an analyst model and specialized
search tools until it can produce a launch-intelligence memo. The result is
written to `cookbook-output/langgraph-launch-intelligence.md`.

## Original source

Inspired by LangGraph's official
[Agentic RAG notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb),
particularly its explicit graph state and tool-routing loop.

This adaptation was **enhanced with SerpApi** by giving the graph separate web,
news, and shopping tools, allowing it to reconcile product claims, current
coverage, and live price signals.

## Run

Set `SERPAPI_API_KEY` and `OPENAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --with . --with langgraph --with langchain-openai \
  cookbook/langgraph/main.py
```

Optional environment variables: `OPENAI_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.

