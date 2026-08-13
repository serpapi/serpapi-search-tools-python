from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LiveParameterCase:
    id: str
    factory: str
    default_params: dict[str, Any]
    arguments: dict[str, Any]
    result_key: str
    result_limit: int = 10


LIVE_PARAMETER_CASES = (
    LiveParameterCase(
        id="readme-web-localized",
        factory="web_search",
        default_params={"hl": "en", "gl": "us"},
        result_limit=3,
        arguments={"query": "Python packaging"},
        result_key="organic_results",
    ),
    LiveParameterCase(
        id="readme-news-localized",
        factory="news_search",
        default_params={"hl": "en", "gl": "us"},
        result_limit=20,
        arguments={"query": "Python releases"},
        result_key="news_results",
    ),
    LiveParameterCase(
        id="examples-maps-localized",
        factory="maps_search",
        default_params={"hl": "en", "gl": "us"},
        arguments={"query": "coffee", "location": "Austin, Texas", "zoom": 12},
        result_key="local_results",
    ),
    LiveParameterCase(
        id="guide-images-safe",
        factory="images_search",
        default_params={"safe": "active"},
        arguments={"query": "latte art"},
        result_key="images_results",
    ),
    LiveParameterCase(
        id="examples-shopping-bounded",
        factory="shopping_search",
        default_params={},
        result_limit=3,
        arguments={"query": "coffee grinder"},
        result_key="shopping_results",
    ),
    LiveParameterCase(
        id="guide-videos-simple",
        factory="videos_search",
        default_params={"hl": "en", "gl": "us"},
        arguments={"query": "latte art"},
        result_key="video_results",
    ),
    LiveParameterCase(
        id="guide-hotels-currency",
        factory="hotels_search",
        default_params={"currency": "USD", "gl": "us"},
        arguments={
            "query": "hotels in Austin",
            "check_in_date": "$check_in",
            "check_out_date": "$check_out",
        },
        result_key="properties",
    ),
    LiveParameterCase(
        id="guide-flights-currency",
        factory="flights_search",
        default_params={"currency": "USD", "hl": "en"},
        arguments={
            "departure_id": "LAX",
            "arrival_id": "AUS",
            "outbound_date": "$check_in",
        },
        result_key="other_flights",
    ),
    LiveParameterCase(
        id="guide-explore-currency",
        factory="travel_explore_search",
        default_params={"currency": "USD", "gl": "us"},
        arguments={
            "departure_id": "AUS",
            "outbound_date": "$check_in",
            "return_date": "$check_out",
        },
        result_key="destinations",
    ),
)
