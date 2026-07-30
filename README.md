# serpapi-search-tools

[![PyPI version](https://img.shields.io/pypi/v/serpapi-search-tools.svg)](https://pypi.org/project/serpapi-search-tools/)
[![CI](https://github.com/serpapi/serpapi-search-tools-python/actions/workflows/ci.yml/badge.svg)](https://github.com/serpapi/serpapi-search-tools-python/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/serpapi-search-tools.svg)](https://pypi.org/project/serpapi-search-tools/)
[![License: MIT](https://img.shields.io/pypi/l/serpapi-search-tools.svg)](https://github.com/serpapi/serpapi-search-tools-python/blob/main/LICENSE)

Give Python agents live web, news, maps, image, shopping, video, hotel, and
flight search through small typed tools.

`serpapi-search-tools` connects [SerpApi](https://serpapi.com) to popular Python
agent SDKs. It provides a separate constructor for each kind of search, so you
can give an agent only the capabilities it needs:

```python
from serpapi_search_tools import maps_search, news_search, web_search

tools = [
    web_search(),
    news_search(),
    maps_search(),
]
```

When one supported agent SDK is installed, the package detects it and creates
tools ready for that SDK. Each search tool asks for the information it needs:
web search uses a query, hotel search requires stay dates, and flight search
uses airports and travel dates. Inputs are validated before a request is sent
to SerpApi, helping prevent failed searches and unnecessary API usage.

Use this package when an agent needs current public information or specialized
search results. You may not need it if your application never searches outside
its own data, or if a one-off HTTP request is enough and you do not need
agent-tool schemas.

## Install

If your agent SDK is already installed, add only the base package:

```bash
pip install serpapi-search-tools
```

If you want this package to install a compatible agent SDK too, choose its
extra. For example:

```bash
pip install "serpapi-search-tools[openai-agents]"
```

Extras are available for all supported SDKs listed below.

Set a SerpApi key:

```bash
export SERPAPI_API_KEY="your-key"
```

`SERPAPI_KEY` is also supported. A directly supplied `api_key=` takes
precedence over environment variables.

## Quickstart: direct Python

Use `provider="function"` when you do not need an agent framework:

```python
import json

from serpapi_search_tools import web_search

search = web_search(
    provider="function",
    allowed_engines=["google_light", "bing"],
    default_params={"num": 3, "hl": "en", "gl": "us"},
)

result = json.loads(search(query="Python packaging"))
print(result["organic_results"])
```

The callable returns compact JSON text containing only the result families an
agent normally needs. Google search keeps answer boxes, knowledge graphs, AI
overviews, and up to five organic results. Google Light also keeps its related
questions, related searches, and top stories. Specialized tools keep up to five
primary results for their vertical. A successful response without a recognized
result family returns a bounded `no_results` status instead of `{}`.

Request the untouched SerpApi response when application code needs metadata,
pagination, filters, or another auxiliary section:

```python
from serpapi_search_tools import SearchResultMode, web_search

full_search = web_search(mode=SearchResultMode.FULL)
```

## Quickstart: an agent with several search capabilities

This example assumes the `openai-agents` extra is the only supported SDK in the
environment, so automatic detection needs no `provider=` argument:

```python
from agents import Agent, Runner

from serpapi_search_tools import maps_search, news_search, web_search

agent = Agent(
    name="research-agent",
    instructions="Use the most specific search tool for the request.",
    tools=[
        web_search(),
        news_search(),
        maps_search(),
    ],
)

result = Runner.run_sync(agent, "Find recent reporting and local events about coffee in Austin")
print(result.final_output)
```

Automatic detection is the normal path when one supported SDK is installed. If
multiple supported SDK families share an environment, automatic detection
raises an actionable error; choose one explicitly, for example
`web_search(provider="openai-agents")`. LangGraph and LangChain count as one
adapter family. When no supported SDK is installed, automatic detection falls
back to a plain callable.

For a step-by-step explanation, keys, customization, and troubleshooting, read
the [detailed quickstart](https://serpapi.github.io/serpapi-search-tools-python/user-guide/quickstart.html).

## Choose an example by task

The repository includes small SDK installation checks and fuller scenarios that
show how applications compose and consume search tools:

| Goal | Example | What it demonstrates |
| --- | --- | --- |
| Research across several verticals | [`direct_multi_search.py`](https://github.com/serpapi/serpapi-search-tools-python/blob/main/examples/direct_multi_search.py) | Web, news, maps, and shopping with bounded summaries |
| Compare marketplaces | [`direct_marketplace_comparison.py`](https://github.com/serpapi/serpapi-search-tools-python/blob/main/examples/direct_marketplace_comparison.py) | Google Shopping, Amazon, Walmart, and eBay normalized into one shape |
| Configure regional tools | [`direct_regioned_search.py`](https://github.com/serpapi/serpapi-search-tools-python/blob/main/examples/direct_regioned_search.py) | Separate names and `gl`/`hl` defaults for US and German search |
| Avoid duplicate requests | [`direct_cached_search.py`](https://github.com/serpapi/serpapi-search-tools-python/blob/main/examples/direct_cached_search.py) | A custom client that caches identical calls and logs safe parameters |
| Plan a structured trip | [`openai_agents_travel_planner.py`](https://github.com/serpapi/serpapi-search-tools-python/blob/main/examples/openai_agents_travel_planner.py) | Explore, flights, and hotels exposed together to OpenAI Agents |

All direct scenarios need only a SerpApi key. Agent scenarios additionally need
the model-provider key used by that script. See
[`examples/README.md`](https://github.com/serpapi/serpapi-search-tools-python/blob/main/examples/README.md)
for exact commands and the complete SDK matrix.

Compact results are the package default, so multi-tool agents do not need
application-owned response wrappers. This avoids sending large maps, shopping,
or travel payloads back to the model. Use `mode=SearchResultMode.FULL` only
when code outside the model context needs the complete response.

## Build a complete agent from the cookbook

The
[`cookbook/`](https://github.com/serpapi/serpapi-search-tools-python/tree/main/cookbook)
directory contains one outcome-driven agent for every supported SDK. Each entry
starts from an official provider example, clearly credits the original source,
and explains how the workflow was enhanced with SerpApi.

The cookbook includes complete market research, company intelligence, trip
planning, newsletter, purchase research, source verification, and retail
location workflows. Every agent loads the same repository-root `.env` format,
writes a Markdown artifact, and has a standalone `uv run` command.

Start with the
[cookbook index](https://serpapi.github.io/serpapi-search-tools-python/docs/cookbook/)
or copy the environment template:

```bash
cp cookbook/sample.env .env
```

## Choose the right tool

| Constructor | SerpApi engine(s) | Required search inputs |
| --- | --- | --- |
| `web_search` | `google`, `google_light`, `bing`, `yahoo`, `duckduckgo` | `query` |
| `news_search` | `google_news` | `query` |
| `maps_search` | `google_maps` | `query` |
| `images_search` | `google_images` | `query` |
| `shopping_search` | `google_shopping`, `amazon`, `walmart`, `ebay` | `query` |
| `videos_search` | `youtube` | `query` |
| `hotels_search` | `google_hotels` | `query`, `check_in_date`, `check_out_date` |
| `flights_search` | `google_flights` | `departure_id`, `arrival_id`, `outbound_date` |
| `travel_explore_search` | `google_travel_explore` | `departure_id` |

Only `web_search` and `shopping_search` expose an `engine` choice to the model.
Every other constructor fixes the engine and presents a schema tailored to that
search intent.

### General web

```python
from serpapi_search_tools import WebSearchEngine, web_search

tool = web_search(
    allowed_engines=[WebSearchEngine.GOOGLE_LIGHT, WebSearchEngine.BING],
    default_engine=WebSearchEngine.GOOGLE_LIGHT,
)
```

`google_light` is the default because it is a fast general-purpose web search.
Yahoo is routed through its native `p` query parameter; the other supported web
engines use `q`.

### News, maps, images, and videos

```python
from serpapi_search_tools import images_search, maps_search, news_search, videos_search

tools = [
    news_search(),
    maps_search(),
    images_search(),
    videos_search(),
]
```

`news_search` supports keyword searches in Google News. `maps_search` searches
Google Maps and accepts optional `location`, `zoom` (`3` through `30`), and
`nearby` fields. Place details, reviews, and directions use different SerpApi
APIs and are not part of this search tool.

### Shopping

```python
from serpapi_search_tools import ShoppingSearchEngine, shopping_search

tool = shopping_search(
    allowed_engines=[
        ShoppingSearchEngine.GOOGLE_SHOPPING,
        ShoppingSearchEngine.AMAZON,
        ShoppingSearchEngine.WALMART,
        ShoppingSearchEngine.EBAY,
    ],
)
```

The package routes one human `query` to each marketplace's native field:
Google Shopping uses `q`, Amazon uses `k`, Walmart uses `query`, and eBay uses
`_nkw`.

### Hotels

```python
from serpapi_search_tools import hotels_search

hotels = hotels_search(provider="function")
result = hotels(
    query="hotels in Kyoto",
    check_in_date="2026-08-01",
    check_out_date="2026-08-04",
    adults=2,
    children=1,
    children_ages=[8],
)
```

Hotel dates use `YYYY-MM-DD`. Checkout must be after check-in. When `children`
is nonzero, provide exactly one age from 1 through 17 per child.

### Flights

```python
from serpapi_search_tools import TravelClass, flights_search

flights = flights_search(provider="function")
result = flights(
    departure_id="LAX",
    arrival_id="AUS",
    outbound_date="2026-08-01",
    return_date="2026-08-04",
    travel_class=TravelClass.BUSINESS,
    adults=1,
)
```

`flights_search` requires an origin, destination, and outbound date. Omitting
`return_date` creates a one-way request; including it creates a round trip.
Multi-city searches are not currently supported.

### Explore destinations

```python
from serpapi_search_tools import travel_explore_search

explore = travel_explore_search(provider="function")
result = explore(
    departure_id="JFK",
    arrival_area_id="/m/02j9z",
)
```

Travel Explore requires only a departure identifier. It can also accept an
arrival identifier or area, fixed outbound/return dates, cabin class, and
passenger counts. These travel tools send their route and date fields directly
to the matching SerpApi endpoint.

## Set advanced parameters in application code

Use `default_params` for documented SerpApi settings that should stay under
your application's control, such as locale, currency, safe search, or result
count. The agent continues to supply only the inputs described by its search
tool.

Applications can supply documented advanced options at construction time:

```python
tool = news_search(
    default_params={"hl": "en", "gl": "us"},
)
```

Typed fields and the constructor-controlled engine override colliding entries
in `default_params`. Known incompatible combinations are rejected locally, such
as Google News query plus topic tokens, Amazon keyword search plus `node`, or
flight airline include plus exclude filters.
For a multi-engine tool, the same defaults are sent to every allowed engine.
Use only parameters shared by those engines, or create separate tool instances
when each engine needs different defaults.
Reserved keys (`api_key`, `async`, `engine`, and `output`) are rejected in
`default_params`; use the constructor options documented below instead.

## Handle search failures

The built-in client raises `SerpApiSearchError` when SerpApi rejects or cannot
complete a request. Its message is sanitized so an API key embedded in an
upstream request URL is replaced with `[REDACTED]`:

```python
from serpapi_search_tools import SerpApiSearchError, web_search

search = web_search(provider="function")
try:
    result = search(query="Python packaging")
except SerpApiSearchError as exc:
    print(f"Search failed: {exc}")
```

Local input errors such as invalid dates or incompatible parameters remain
`ValueError`, so applications can distinguish validation from provider and
transport failures.

## Common factory options

Every constructor accepts:

| Option | Purpose |
| --- | --- |
| `provider` | Defaults to `"auto"`; select an SDK explicitly in multi-SDK environments or use `"function"` for a plain callable |
| `include_examples` | Include or omit a short example in the model description |
| `api_key` | Explicit SerpApi key |
| `client` | Custom object with `search(params)` for caching, interception, or tests |
| `default_params` | Application-controlled SerpApi options |
| `timeout` | Timeout passed to the SerpApi SDK client |
| `name` | Tool name presented to the model |
| `mode` | Result detail level: `"compact"` (default) or `"full"` |

`web_search` and `shopping_search` additionally accept `allowed_engines` and
`default_engine`. The tool offers only the engine values you configured.

## Supported agent SDKs

| Provider value | Install extra | Returned tool |
| --- | --- | --- |
| `openai-agents` | `openai-agents` | OpenAI Agents `FunctionTool` |
| `pydantic-ai` | `pydantic-ai` | Pydantic AI `Tool` |
| `langchain` | `langchain` | LangChain `StructuredTool` |
| `langgraph` | `langgraph` | LangChain-compatible structured tool |
| `crewai` | `crewai` | CrewAI `BaseTool` |
| `llamaindex` | `llamaindex` | LlamaIndex `FunctionTool` |
| `claude-agent-sdk` | `claude-agent-sdk` | Claude SDK MCP tool |
| `microsoft-agent-framework` | `microsoft-agent-framework` | Microsoft Agent Framework `FunctionTool` |
| `autogen` | `autogen` | AutoGen `FunctionTool` |
| `haystack` | `haystack` | Haystack `Tool` |
| `semantic-kernel` | `semantic-kernel` | Semantic Kernel function |
| `agno` | `agno` | Agno `Function` |
| `smolagents` | `smolagents` | smolagents `Tool` |
| `google-adk` | `google-adk` | Google ADK `FunctionTool` |

Agent SDK dependencies are optional and loaded only when you use them. The base
package depends only on the official `serpapi` Python client.

Microsoft Agent Framework is Microsoft's recommended successor for AutoGen and
Semantic Kernel agent applications. Both older adapters remain supported for
existing projects. CrewAI is supported on Python 3.10 through 3.13; the base
package and other compatible adapters support Python 3.14.

## Supported engine documentation

- [Google Search](https://serpapi.com/search-api)
- [Google Light](https://serpapi.com/google-light-api)
- [Bing](https://serpapi.com/bing-search-api)
- [Yahoo](https://serpapi.com/yahoo-search-api)
- [DuckDuckGo](https://serpapi.com/duckduckgo-search-api)
- [Google News](https://serpapi.com/google-news-api)
- [Google Maps](https://serpapi.com/google-maps-api)
- [Google Images](https://serpapi.com/google-images-api)
- [Google Shopping](https://serpapi.com/google-shopping-api)
- [Amazon](https://serpapi.com/amazon-search-api)
- [Walmart](https://serpapi.com/walmart-search-api)
- [eBay](https://serpapi.com/ebay-search-api)
- [YouTube](https://serpapi.com/youtube-search-api)
- [Google Hotels](https://serpapi.com/google-hotels-api)
- [Google Flights](https://serpapi.com/google-flights-api)
- [Google Travel Explore](https://serpapi.com/google-travel-explore-api)

SerpApi's complete documentation index is available in
[`llms.txt`](https://serpapi.com/llms.txt).

## More guides

- [Introduction](https://serpapi.github.io/serpapi-search-tools-python/user-guide/introduction.html)
- [Quickstart](https://serpapi.github.io/serpapi-search-tools-python/user-guide/quickstart.html)
- [Choose a search tool](https://serpapi.github.io/serpapi-search-tools-python/user-guide/search_tools.html)
- [Usage and composition](https://serpapi.github.io/serpapi-search-tools-python/user-guide/usage.html)
- [Configuration](https://serpapi.github.io/serpapi-search-tools-python/user-guide/configuration.html)
- [Recipes](https://serpapi.github.io/serpapi-search-tools-python/user-guide/recipes.html)
- [Agent SDKs](https://serpapi.github.io/serpapi-search-tools-python/user-guide/frameworks.html)
- [Agent cookbook](https://serpapi.github.io/serpapi-search-tools-python/docs/cookbook/)
- [Runnable examples](https://serpapi.github.io/serpapi-search-tools-python/sdk-examples/)
- [Testing and contributing](https://serpapi.github.io/serpapi-search-tools-python/user-guide/testing.html)
