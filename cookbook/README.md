# Agent cookbook

Choose the SDK you use and run a complete agent with live SerpApi search.
Every entry includes setup instructions, an editable prompt, a runnable agent,
and a Markdown report you can inspect.

| SDK | Agent | SerpApi capabilities |
| --- | --- | --- |
| [LangChain](langchain/) | Deep market research brief | Web and news |
| [LangGraph](langgraph/) | Product-launch intelligence graph | Web, news, and shopping |
| [CrewAI](crewai/) | Collaborative trip planner | Flights, hotels, and maps |
| [LlamaIndex](llamaindex/) | Remote-work destination brief | Travel explore, web, and maps |
| [OpenAI Agents](openai-agents/) | Managed research report | Web and news |
| [Claude Agent SDK](claude-agent-sdk/) | Source-verification memo | Web and news |
| [Pydantic AI](pydantic-ai/) | Visual location scout | Images, maps, and web |
| [Microsoft Agent Framework](microsoft-agent-framework/) | Technology due-diligence memo | Web and news |
| [AutoGen](autogen/) | Company intelligence memo | Web and news |
| [Haystack](haystack/) | Weekly industry newsletter | Web and news |
| [Semantic Kernel](semantic-kernel/) | Plan-and-execute competitor brief | Web and news |
| [Agno](agno/) | Market research report | Web, news, and shopping |
| [smolagents](smolagents/) | Purchase research assistant | Shopping, images, and videos |
| [Google ADK](google-adk/) | Retail location strategy | Maps, web, and news |

## Configure the environment

From the repository root, copy the sample environment and fill in the keys for
the cookbook entry you want to run:

```bash
cp cookbook/sample.env .env
```

Every entry needs `SERPAPI_API_KEY` (or `SERPAPI_KEY`) plus the model-provider
key named in its README.

Set `COOKBOOK_PROMPT` to replace an entry's default brief. Generated reports go
to `cookbook-output/` by default; set `COOKBOOK_OUTPUT_DIR` to change that.

## Run an entry

Each entry is a standalone script. For example:

```bash
uv run --isolated --no-project --with 'serpapi-search-tools[langchain]' --with python-dotenv --with langchain-openai \
  cookbook/langchain/main.py
```

`--no-project` installs the published package instead of the repository's local
checkout.

Open the entry's README for its exact command and source attribution.
