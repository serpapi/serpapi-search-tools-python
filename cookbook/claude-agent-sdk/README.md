# Claude Agent SDK source-verification memo

This agent exposes SerpApi tools through an in-process SDK MCP server and uses
them to verify a broad public-policy claim. It writes the final memo to
`cookbook-output/claude-source-verification.md`.

## Original source

Inspired by Anthropic's official
[custom tools through SDK MCP servers](https://github.com/anthropics/claude-agent-sdk-python#custom-tools-as-in-process-sdk-mcp-servers)
example.

This adaptation was **enhanced with SerpApi** by replacing the demonstration
calculator-style tool with live web and news search and by turning the workflow
into an evidence reconciliation task.

## Run

Set `SERPAPI_API_KEY` and `ANTHROPIC_API_KEY` in the repository-root `.env`,
then:

```bash
uv run --isolated --no-project --with 'serpapi-search-tools[claude-agent-sdk]' --with python-dotenv cookbook/claude-agent-sdk/main.py
```

Optional environment variables: `ANTHROPIC_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.

