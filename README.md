# serpapi-search-tools

[![PyPI version](https://img.shields.io/pypi/v/serpapi-search-tools.svg)](https://pypi.org/project/serpapi-search-tools/)
[![CI](https://github.com/serpapi/serpapi-search-tools-python/actions/workflows/ci.yml/badge.svg)](https://github.com/serpapi/serpapi-search-tools-python/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/serpapi-search-tools.svg)](https://pypi.org/project/serpapi-search-tools/)
[![License: MIT](https://img.shields.io/pypi/l/serpapi-search-tools.svg)](https://github.com/serpapi/serpapi-search-tools-python/blob/main/LICENSE)

[Read the full documentation](https://serpapi.github.io/serpapi-search-tools-python/)
for guides, SDK examples, recipes, and the API reference.

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
uses airports and travel dates.

## Supported agent SDKs

| SDK | Install extra / provider | Returned tool |
| --- | --- | --- |
| OpenAI Agents SDK | `openai-agents` | OpenAI Agents `FunctionTool` |
| Pydantic AI | `pydantic-ai` | Pydantic AI `Tool` |
| LangChain | `langchain` | LangChain `StructuredTool` |
| LangGraph | `langgraph` | LangChain-compatible structured tool |
| CrewAI | `crewai` | CrewAI `BaseTool` |
| LlamaIndex | `llamaindex` | LlamaIndex `FunctionTool` |
| Claude Agent SDK | `claude-agent-sdk` | Claude SDK MCP tool |
| Microsoft Agent Framework | `microsoft-agent-framework` | Microsoft Agent Framework `FunctionTool` |
| AutoGen | `autogen` | AutoGen `FunctionTool` |
| Haystack | `haystack` | Haystack `Tool` |
| Semantic Kernel | `semantic-kernel` | Semantic Kernel function |
| Agno | `agno` | Agno `Function` |
| smolagents | `smolagents` | smolagents `Tool` |
| Google ADK | `google-adk` | Google ADK `FunctionTool` |


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

Extras are available for all supported SDKs listed above.

Set a SerpApi key:

```bash
export SERPAPI_API_KEY="your-key"
```

`SERPAPI_KEY` is also supported. A directly supplied `api_key=` takes
precedence over environment variables.

## Quickstart: automatic SDK detection

This quickstart uses OpenAI Agents SDK to demonstrate automatic detection. It
assumes the SDK is already installed in your environment (install it with
`pip install openai-agents` if needed). Then add the base package:

```bash
pip install serpapi-search-tools
```

With one supported SDK installed, create the tool without any configuration.
The package detects OpenAI Agents SDK and returns its native `FunctionTool`.
This example also expects the `OPENAI_API_KEY` used by your agent.

```python
from agents import Agent, Runner

from serpapi_search_tools import web_search

agent = Agent(
    name="research-agent",
    instructions="Use web search when the answer needs current information.",
    tools=[web_search()],
)

result = Runner.run_sync(
    agent,
    "Find three recent Python packaging changes and explain why they matter.",
)
print(result.final_output)
```

## Quickstart: explicit LangChain provider

Install the LangChain extra and the model backend used by this example:

```bash
pip install "serpapi-search-tools[langchain]" langchain-openai
```

The `langchain` extra installs a compatible LangChain version.
`langchain-openai` provides this example's model integration; replace it with
the backend your LangChain application uses. With `langchain-openai`, set
`OPENAI_API_KEY` before running the agent.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from serpapi_search_tools import maps_search, news_search, web_search

agent = create_agent(
    model=ChatOpenAI(model="gpt-5.4-mini", temperature=0),
    tools=[
        web_search(provider="langchain"),
        news_search(provider="langchain"),
        maps_search(provider="langchain"),
    ],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Research coffee culture in Austin using current reporting, "
                    "local places, and general web sources."
                ),
            }
        ]
    }
)
print(result["messages"][-1].content)
```

When LangChain is the only supported SDK installed, these constructors are
also auto-detected, so `web_search()` is enough. Supplying
`provider="langchain"` explicitly is useful when several supported SDKs share
an environment or when you want the integration choice to be visible in code.

For a step-by-step explanation, keys, customization, and troubleshooting, read
the [detailed quickstart](https://serpapi.github.io/serpapi-search-tools-python/user-guide/quickstart.html).

Browse the [runnable examples](https://github.com/serpapi/serpapi-search-tools-python/tree/main/examples)
for focused integrations or the [agent cookbook](https://serpapi.github.io/serpapi-search-tools-python/docs/cookbook/)
for complete, task-oriented agents built with every supported SDK.

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

### General web

```python
from serpapi_search_tools import WebSearchEngine, web_search

tool = web_search(
    allowed_engines=[WebSearchEngine.GOOGLE_LIGHT, WebSearchEngine.BING],
    default_engine=WebSearchEngine.GOOGLE_LIGHT,
)
```

SerpApi supports multiple general web search engines, including Google Light,
Google, Bing, Yahoo, and DuckDuckGo. Google Light is the default because it
provides fast, general-purpose web results. Use `allowed_engines` to choose
which engines are available and `default_engine` to select the initial one.

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
complete a request:

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
- [Runnable examples](https://serpapi.github.io/serpapi-search-tools-python/docs/sdk-examples/)
