from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordingSearchClient:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def search(self, params: dict[str, Any]) -> Mapping[str, Any] | str:
        call = dict(params)
        self.calls.append(call)
        primary_key = {
            "google_news": "news_results",
            "google_maps": "local_results",
            "google_images": "images_results",
            "google_shopping": "shopping_results",
            "youtube": "video_results",
            "google_hotels": "properties",
            "google_flights": "best_flights",
            "google_travel_explore": "destinations",
        }.get(str(call["engine"]), "organic_results")
        if call.get("output") == "md":
            heading = {
                "news_results": "News Results",
                "local_results": "Local Results",
                "images_results": "Images Results",
                "shopping_results": "Shopping Results",
                "video_results": "Video Results",
                "properties": "Properties",
                "best_flights": "Best Flights",
                "destinations": "Destinations",
            }.get(primary_key, "Organic Results")
            return (
                "---\nsearch_metadata:\n  status: Success\n---\n\n"
                f"## {heading}\n\n"
                "| Title | Link |\n"
                "| --- | --- |\n"
                f"| Fake result for {call} | https://example.com/result |\n"
            )
        return {
            "search_metadata": {"status": "Success"},
            "search_parameters": call,
            primary_key: [
                {"title": f"Fake result for {call}", "link": "https://example.com/result"}
            ],
        }
