# Run: uv run --with . examples/direct_travel.py
"""Use the three structured travel tools directly, with no model involved."""

from __future__ import annotations

import json
from datetime import date, timedelta

from dotenv import load_dotenv

from serpapi_search_tools import flights_search, hotels_search, travel_explore_search

load_dotenv()


def main() -> None:
    outbound = date.today() + timedelta(days=60)
    returning = outbound + timedelta(days=4)

    hotels = hotels_search(provider="function")
    flights = flights_search(provider="function")
    explore = travel_explore_search(provider="function")

    hotel_result = hotels(
        query="hotels in Austin",
        check_in_date=outbound.isoformat(),
        check_out_date=returning.isoformat(),
        adults=2,
    )
    flight_result = flights(
        departure_id="LAX",
        arrival_id="AUS",
        outbound_date=outbound.isoformat(),
        return_date=returning.isoformat(),
    )
    explore_result = explore(departure_id="JFK")

    for label, result_keys, encoded in (
        ("hotels", ("properties",), hotel_result),
        ("flights", ("best_flights", "other_flights"), flight_result),
        ("explore", ("destinations",), explore_result),
    ):
        result = json.loads(encoded)
        count = sum(
            len(result.get(key, [])) for key in result_keys if isinstance(result.get(key), list)
        )
        print(label, count)


if __name__ == "__main__":
    main()
