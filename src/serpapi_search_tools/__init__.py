"""Intent-specific SerpApi tools for Python agent SDKs."""

from serpapi_search_tools._query_tools import (
    ShoppingSearchEngine,
    WebSearchEngine,
    images_search,
    maps_search,
    news_search,
    shopping_search,
    videos_search,
    web_search,
)
from serpapi_search_tools._shared import SerpApiSearchError
from serpapi_search_tools._travel_tools import (
    TravelClass,
    flights_search,
    hotels_search,
    travel_explore_search,
)

_SUPPORTED_ENGINES_BY_TOOL: dict[str, tuple[str, ...]] = {
    "web_search": tuple(engine.value for engine in WebSearchEngine),
    "news_search": ("google_news",),
    "maps_search": ("google_maps",),
    "images_search": ("google_images",),
    "shopping_search": tuple(engine.value for engine in ShoppingSearchEngine),
    "videos_search": ("youtube",),
    "hotels_search": ("google_hotels",),
    "flights_search": ("google_flights",),
    "travel_explore_search": ("google_travel_explore",),
}

__all__ = [
    "SerpApiSearchError",
    "ShoppingSearchEngine",
    "TravelClass",
    "WebSearchEngine",
    "flights_search",
    "hotels_search",
    "images_search",
    "maps_search",
    "news_search",
    "shopping_search",
    "travel_explore_search",
    "videos_search",
    "web_search",
]
