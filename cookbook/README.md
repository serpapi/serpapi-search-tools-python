# Agent cookbook

This cookbook contains one outcome-driven agent for every SDK supported by
`serpapi-search-tools`. Each entry starts from an official provider example and
enhances the workflow with live SerpApi search tools.

These are fuller applications than the small scripts in `examples/`: each
cookbook entry has a real objective, writes a Markdown artifact, and documents
its upstream inspiration.

| SDK | Agent | SerpApi capabilities |
| --- | --- | --- |
| LangChain | Deep market research brief | Web and news |
| LangGraph | Product-launch intelligence graph | Web, news, and shopping |
| CrewAI | Collaborative trip planner | Flights, hotels, and maps |
| LlamaIndex | Remote-work destination brief | Travel explore, web, and maps |
| OpenAI Agents | Managed research report | Web and news |
| Claude Agent SDK | Source-verification memo | Web and news |
| Pydantic AI | Visual location scout | Images, maps, and web |
| Microsoft Agent Framework | Technology due-diligence memo | Web and news |
| AutoGen | Company intelligence memo | Web and news |
| Haystack | Weekly industry newsletter | Web and news |
| Semantic Kernel | Plan-and-execute competitor brief | Web and news |
| Agno | Market research report | Web, news, and shopping |
| smolagents | Purchase research assistant | Shopping, images, and videos |
| Google ADK | Retail location strategy | Maps, web, and news |

## Configure the environment

From the repository root, copy the sample environment and fill in the keys for
the cookbook entry you want to run:

```bash
cp cookbook/sample.env .env
```

Every entry needs `SERPAPI_API_KEY` (or `SERPAPI_KEY`) plus the model-provider
key named in its README. All scripts load the repository-root `.env` file in the
same way as the existing runnable examples.

Set `COOKBOOK_PROMPT` to replace an entry's default brief. Generated reports go
to `cookbook-output/` by default; set `COOKBOOK_OUTPUT_DIR` to change that.

## Run an entry

Each entry is a standalone script. For example:

```bash
uv run --with . --with langchain --with langchain-openai \
  cookbook/langchain/main.py
```

Open the entry's README for its exact command and source attribution.
