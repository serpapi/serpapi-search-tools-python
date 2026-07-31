# Runnable examples

Start with the direct examples. They need no agent SDK extra. These commands use
the published package and add `python-dotenv` so the scripts can load `.env`:

```bash
uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_search.py
uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_travel.py
uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_multi_search.py
uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_marketplace_comparison.py
uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_regioned_search.py
uv run --isolated --no-project --with serpapi-search-tools --with python-dotenv examples/direct_cached_search.py
```

Every script loads `.env` before reading credentials. Copy `examples/sample.env`
to `.env`, or export values in your shell. You always need `SERPAPI_API_KEY` or
`SERPAPI_KEY`; agent examples also need their model provider key.

| Script | Scenario | Extra | Search tools demonstrated |
| --- | --- | --- | --- |
| `direct_search.py` | First direct call | none | Web and news |
| `direct_travel.py` | Structured travel calls | none | Hotels, flights, and travel explore |
| `direct_multi_search.py` | Multi-vertical research | none | Web, news, maps, and shopping |
| `direct_marketplace_comparison.py` | Normalize marketplace results | none | Google Shopping, Amazon, Walmart, and eBay |
| `direct_regioned_search.py` | Named regional tools | none | Two Google Light configurations |
| `direct_cached_search.py` | Cache identical requests | none | Web with a custom client |
| `openai_agents_openai.py` | Agent chooses several verticals | `openai-agents` | Web, news, maps, and shopping |
| `openai_agents_travel_planner.py` | Typed agent travel planning | `openai-agents` | Hotels, flights, and travel explore |
| `pydantic_ai_openai.py` | Visual research | `pydantic-ai` | Images and web |
| `langchain_grok.py` | Local discovery | `langchain` + model backend | Maps and web |
| `langgraph_openai.py` | Stateful research | `langgraph` + model backend | Web and news |
| `crewai_grok.py` | Product research | `crewai` | Shopping |
| `microsoft_agent_framework_openai.py` | Current research | `microsoft-agent-framework` | News and web |
| `autogen_openai.py` | Current research | `autogen` | News and web |
| `haystack_openai.py` | Local discovery | `haystack` | Maps and web |
| `llamaindex_openai.py` | Destination research | `llamaindex` + model backend | Travel explore and web |
| `agno_grok.py` | Product research | `agno` | Web and shopping |
| `smolagents_openai.py` | Video discovery | `smolagents` | Videos |
| `semantic_kernel_openai.py` | Local discovery | `semantic-kernel` | Maps and web |
| `claude_agent_sdk_sonnet.py` | MCP search tools | `claude-agent-sdk` | News and web |
| `google_adk_gemini.py` | Product research | `google-adk` | Shopping and web |

Run an agent example with its extra:

```bash
uv run --isolated --no-project --with 'serpapi-search-tools[openai-agents]' --with python-dotenv examples/openai_agents_openai.py
uv run --isolated --no-project --with 'serpapi-search-tools[openai-agents]' --with python-dotenv examples/openai_agents_travel_planner.py
uv run --isolated --no-project --with 'serpapi-search-tools[langchain]' --with python-dotenv --with langchain-openai examples/langchain_grok.py
uv run --isolated --no-project --with 'serpapi-search-tools[langgraph]' --with python-dotenv --with langchain-openai examples/langgraph_openai.py
uv run --isolated --no-project --with 'serpapi-search-tools[llamaindex]' --with python-dotenv --with llama-index-llms-openai \
  examples/llamaindex_openai.py
uv run --isolated --no-project --with 'serpapi-search-tools[microsoft-agent-framework]' --with python-dotenv examples/microsoft_agent_framework_openai.py
uv run --isolated --no-project --with 'serpapi-search-tools[claude-agent-sdk]' --with python-dotenv examples/claude_agent_sdk_sonnet.py
uv run --isolated --no-project --with 'serpapi-search-tools[google-adk]' --with python-dotenv examples/google_adk_gemini.py
```

`--no-project` keeps these commands independent of the repository's local
package and development environment.

The examples default to `gpt-5.4-mini`, `claude-sonnet-5`,
`gemini-flash-lite-latest`, and `grok-4.5`. Override the corresponding model
environment variable when needed.

The two multi-tool OpenAI Agents scenarios use `_logging_client.py` to log safe
request metadata and response shape. The package's default compact mode keeps
at most five primary results.
