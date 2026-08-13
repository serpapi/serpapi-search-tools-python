# SerpApi Search Tools

[![PyPI version](https://img.shields.io/pypi/v/serpapi-search-tools.svg?v=1)](https://pypi.org/project/serpapi-search-tools/)
[![CI](https://github.com/serpapi/serpapi-search-tools-python/actions/workflows/ci.yml/badge.svg)](https://github.com/serpapi/serpapi-search-tools-python/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/serpapi-search-tools.svg?v=1)](https://pypi.org/project/serpapi-search-tools/)
[![License: MIT](https://img.shields.io/pypi/l/serpapi-search-tools.svg?v=1)](https://github.com/serpapi/serpapi-search-tools-python/blob/main/LICENSE)

Give Python AI agents live web, news, maps, image, shopping, video, hotel, and flight search with easy-to-use, customizable tools.

[Read the full documentation](https://serpapi.github.io/serpapi-search-tools-python/) for guides, SDK examples, recipes, and the API reference.

The package creates native [SerpApi](https://serpapi.com) tools for popular Python agent SDKs:

```python
from serpapi_search_tools import maps_search, news_search, web_search

tools = [
    web_search(),
    news_search(),
    maps_search(),
]
```

When one supported agent SDK is installed, the package detects it and creates tools ready for that SDK.

## Install

If your agent SDK is already installed, add only the base package:

```bash
pip install serpapi-search-tools
```

If you want this package to install a compatible agent SDK too, choose its extra. For example:

```bash
pip install "serpapi-search-tools[openai-agents]"
```

Extras are available for all supported SDKs listed below.

Set a SerpApi key:

```bash
export SERPAPI_API_KEY="your-key"
```

`SERPAPI_KEY` is also supported. A directly supplied `api_key=` takes precedence over environment variables.

## Supported agent SDKs

| SDK                       | Install extra               | Returned tool                            |
|---------------------------|-----------------------------|------------------------------------------|
| OpenAI Agents SDK         | `openai-agents`             | OpenAI Agents `FunctionTool`             |
| Pydantic AI               | `pydantic-ai`               | Pydantic AI `Tool`                       |
| LangChain                 | `langchain`                 | LangChain `StructuredTool`               |
| LangGraph                 | `langgraph`                 | LangChain-compatible structured tool     |
| CrewAI                    | `crewai`                    | CrewAI `BaseTool`                        |
| LlamaIndex                | `llamaindex`                | LlamaIndex `FunctionTool`                |
| Claude Agent SDK          | `claude-agent-sdk`          | Claude SDK MCP tool                      |
| Microsoft Agent Framework | `microsoft-agent-framework` | Microsoft Agent Framework `FunctionTool` |
| AutoGen                   | `autogen`                   | AutoGen `FunctionTool`                   |
| Haystack                  | `haystack`                  | Haystack `Tool`                          |
| Semantic Kernel           | `semantic-kernel`           | Semantic Kernel function                 |
| Agno                      | `agno`                      | Agno `Function`                          |
| smolagents                | `smolagents`                | smolagents `Tool`                        |
| Google ADK                | `google-adk`                | Google ADK `FunctionTool`                |

## Quickstart: automatic SDK detection

This quickstart uses OpenAI Agents SDK to demonstrate automatic detection. It assumes the SDK is already installed in your environment (install it with `pip install openai-agents` if needed). Then add the base package:

```bash
pip install serpapi-search-tools
```

With one supported SDK installed, create the tool without any configuration. The package detects OpenAI Agents SDK and returns its native `FunctionTool`. This example also expects the `OPENAI_API_KEY` used by your agent.

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

## Quickstart: LangChain

Install the LangChain extra and the model backend used by this example:

```bash
pip install "serpapi-search-tools[langchain]" langchain-openai
```

The `langchain` extra installs a compatible LangChain version. `langchain-openai` provides this example's model integration; replace it with the backend your LangChain application uses. With `langchain-openai`, set `OPENAI_API_KEY` before running the agent.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from serpapi_search_tools import maps_search, news_search, web_search

agent = create_agent(
    model=ChatOpenAI(model="gpt-5.4-mini", temperature=0),
    tools=[
        web_search(),
        news_search(),
        maps_search(),
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

The constructors use automatic SDK detection, just as in the first quickstart. For multi-SDK environments and explicit selection, see [Agent SDKs](https://serpapi.github.io/serpapi-search-tools-python/user-guide/frameworks.html).

For a step-by-step explanation, keys, customization, and troubleshooting, read the [detailed quickstart](https://serpapi.github.io/serpapi-search-tools-python/user-guide/quickstart.html).

Browse the [runnable examples](https://github.com/serpapi/serpapi-search-tools-python/tree/main/examples) for focused integrations or the [agent cookbook](https://serpapi.github.io/serpapi-search-tools-python/docs/cookbook/) for complete, task-oriented agents built with every supported SDK.

## Choose the right tool

| Constructor             | SerpApi engine(s)                                       | Required search inputs                        |
|-------------------------|---------------------------------------------------------|-----------------------------------------------|
| `web_search`            | `google`, `google_light`, `bing`, `yahoo`, `duckduckgo` | `query`                                       |
| `news_search`           | `google_news`                                           | `query`                                       |
| `maps_search`           | `google_maps`                                           | `query`                                       |
| `images_search`         | `google_images`                                         | `query`                                       |
| `shopping_search`       | `google_shopping`, `amazon`, `walmart`, `ebay`          | `query`                                       |
| `videos_search`         | `youtube`                                               | `query`                                       |
| `hotels_search`         | `google_hotels`                                         | `query`, `check_in_date`, `check_out_date`    |
| `flights_search`        | `google_flights`                                        | `departure_id`, `arrival_id`, `outbound_date` |
| `travel_explore_search` | `google_travel_explore`                                 | `departure_id`                                |

### General web

```python
from serpapi_search_tools import WebSearchEngine, web_search

tool = web_search(
    allowed_engines=[WebSearchEngine.GOOGLE_LIGHT, WebSearchEngine.BING],
    default_engine=WebSearchEngine.GOOGLE_LIGHT,
)
```

SerpApi supports multiple general web search engines, including Google Light, Google, Bing, Yahoo, and DuckDuckGo. Google Light is the default because it provides fast, general-purpose web results. Use `allowed_engines` to choose which engines are available and `default_engine` to select the initial one.

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

`news_search` supports keyword searches in Google News. `maps_search` searches Google Maps and accepts optional `location`, `zoom` (`3` through `30`), and `nearby` fields. Use `nearby=True` for “near me” intent with a separate `location`; leave it false when the query already names a city or area. Place details, reviews, and directions use different SerpApi APIs and are not part of this search tool.

`images_search` returns images and their source pages. `videos_search` searches YouTube videos, Shorts, channels, playlists, movies, and categories.

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

Use the same `query` input for every marketplace. The package translates it to the selected engine's request format.

### Travel

```python
from serpapi_search_tools import flights_search, hotels_search, travel_explore_search

travel_tools = [
    hotels_search(),
    flights_search(),
    travel_explore_search(),
]
```

These constructors create hotel, flight, and destination-discovery tools for the detected agent SDK. Read the [Hotels](https://serpapi.github.io/serpapi-search-tools-python/user-guide/hotels_search.html), [Flights](https://serpapi.github.io/serpapi-search-tools-python/user-guide/flights_search.html), and [Travel Explore](https://serpapi.github.io/serpapi-search-tools-python/user-guide/travel_explore_search.html) guides for their date, occupancy, location ID, and trip rules.

## Configure tools

Every constructor accepts:

| Option             | Purpose                                                                                               |
|--------------------|-------------------------------------------------------------------------------------------------------|
| `provider`         | Defaults to `"auto"`; select an SDK explicitly only when multiple supported SDKs share an environment |
| `include_examples` | Include or omit a short example in the model description                                              |
| `api_key`          | Explicit SerpApi key                                                                                  |
| `client`           | Custom object with `search(params)` for caching, interception, or tests                               |
| `default_params`   | Application-controlled SerpApi options                                                                |
| `timeout`          | Timeout passed to the SerpApi SDK client                                                              |
| `name`             | Tool name presented to the model                                                                      |
| `mode`             | Result detail level; compact mode is the default, while full mode keeps supporting sections and all fields on retained results |
| `result_limit`     | Maximum items kept in each result list in either mode; defaults vary by tool; use `None` for all results |

`web_search` and `shopping_search` also accept `allowed_engines` and `default_engine`. The tool offers only the engine values you configure.

Use `default_params` for documented SerpApi settings that should stay under your application's control, such as locale, currency, safe search, or pagination. Use `result_limit` to control how many results the tool returns. The agent continues to supply only the inputs described by its search tool.

```python
tool = news_search(
    default_params={"hl": "en", "gl": "us"},
)
```

Typed tool inputs override matching values in `default_params`, and the constructor always controls `engine`. The package rejects known incompatible combinations, such as a Google News query with a topic token, an Amazon keyword search with `node`, or flight airline include and exclude filters. A multi-engine tool sends the same defaults to every allowed engine, so use parameters shared by those engines or create separate tool instances. Reserved keys (`api_key`, `async`, `engine`, and `output`) are rejected in `default_params`; use the constructor options above instead.

Read [Manage LLM context](https://serpapi.github.io/serpapi-search-tools-python/user-guide/managing_llm_context.html) for compact and full response behavior, default result limits, and unlimited responses.

## Handle search failures

The search runtime raises `SerpApiSearchError` for SerpApi and transport failures. Invalid tool inputs raise `ValueError`. Agent SDKs surface or handle tool errors differently, so use your SDK's normal tool-error mechanism. See [Debugging](https://serpapi.github.io/serpapi-search-tools-python/user-guide/debugging.html) for detailed examples.

## Engine API references

<details>
<summary>Show supported SerpApi engines</summary>

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

</details>

For AI coding agents that need broader SerpApi API context, use [SerpApi's agent-friendly documentation index (`llms.txt`)](https://serpapi.com/llms.txt). It links directly to Markdown API references, including APIs beyond those wrapped by this package.

## More guides

- [Full documentation](https://serpapi.github.io/serpapi-search-tools-python/)
- [SDK examples](https://serpapi.github.io/serpapi-search-tools-python/docs/sdk-examples/)
- [Agent cookbook](https://serpapi.github.io/serpapi-search-tools-python/docs/cookbook/)
