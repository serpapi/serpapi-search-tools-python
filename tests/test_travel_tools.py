import inspect
import re

import pytest

from serpapi_search_tools._travel_tools import (
    TravelClass,
    flights_search,
    hotels_search,
    travel_explore_search,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(self, params: dict[str, object]) -> dict[str, object]:
        self.calls.append(params)
        return {"params": params}


class FailIfCalledClient:
    def search(self, params: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"client must not be called: {params}")


def test_travel_class_is_semantic_public_enum() -> None:
    assert [member.value for member in TravelClass] == [
        "economy",
        "premium_economy",
        "business",
        "first",
    ]


@pytest.mark.parametrize(
    "factory",
    [hotels_search, flights_search, travel_explore_search],
)
def test_model_facing_travel_descriptions_do_not_embed_a_fixed_year(factory) -> None:
    tool = factory(provider="function", client=FakeClient())

    assert re.search(r"\b20\d{2}-\d{2}-\d{2}\b", tool.__doc__ or "") is None


@pytest.mark.parametrize(
    "factory",
    [hotels_search, flights_search, travel_explore_search],
)
def test_travel_tool_signatures_are_closed_and_typed(factory) -> None:
    tool = factory(provider="function", client=FakeClient())
    signature = inspect.signature(tool)

    assert "serpapi_params" not in signature.parameters
    assert signature.return_annotation is str


def test_hotels_search_builds_complete_request() -> None:
    client = FakeClient()
    tool = hotels_search(provider="function", client=client)

    tool(
        query="hotels in Kyoto",
        check_in_date="2026-08-01",
        check_out_date="2026-08-04",
        adults=2,
        children=1,
        children_ages=[8],
    )

    assert client.calls == [
        {
            "engine": "google_hotels",
            "q": "hotels in Kyoto",
            "check_in_date": "2026-08-01",
            "check_out_date": "2026-08-04",
            "adults": 2,
            "children": 1,
            "children_ages": "8",
        }
    ]


def test_flights_search_infers_one_way_type_without_q() -> None:
    client = FakeClient()
    tool = flights_search(
        provider="function",
        client=client,
        default_params={"return_date": "2026-08-04", "type": 1},
    )

    tool(departure_id="LAX", arrival_id="AUS", outbound_date="2026-08-01")

    assert client.calls == [
        {
            "engine": "google_flights",
            "departure_id": "LAX",
            "arrival_id": "AUS",
            "outbound_date": "2026-08-01",
            "type": 2,
            "travel_class": 1,
            "adults": 1,
            "children": 0,
            "infants_in_seat": 0,
            "infants_on_lap": 0,
        }
    ]
    assert "q" not in client.calls[0]


def test_flights_search_infers_round_trip_type() -> None:
    client = FakeClient()
    tool = flights_search(provider="function", client=client)

    tool(
        departure_id="LAX",
        arrival_id="AUS",
        outbound_date="2026-08-01",
        return_date="2026-08-04",
        travel_class="business",
    )

    assert client.calls[0]["type"] == 1
    assert client.calls[0]["travel_class"] == 3


@pytest.mark.parametrize(
    ("factory", "kwargs", "expected_departure", "expected_arrival"),
    [
        (
            flights_search,
            {"departure_id": "lax,jfk", "arrival_id": "aus,/m/02j9z"},
            "LAX,JFK",
            "AUS,/m/02j9z",
        ),
        (
            travel_explore_search,
            {"departure_id": "jfk,/m/02j9z", "arrival_id": "lax"},
            "JFK,/m/02j9z",
            "LAX",
        ),
    ],
)
def test_travel_tools_normalize_airport_codes_without_changing_location_ids(
    factory,
    kwargs: dict[str, object],
    expected_departure: str,
    expected_arrival: str,
) -> None:
    client = FakeClient()
    tool = factory(provider="function", client=client)

    if factory is flights_search:
        tool(**kwargs, outbound_date="2026-08-01")
    else:
        tool(**kwargs)

    assert client.calls[0]["departure_id"] == expected_departure
    assert client.calls[0]["arrival_id"] == expected_arrival


@pytest.mark.parametrize(
    ("outbound_date", "return_date", "expected_type"),
    [(None, None, None), ("2026-08-01", None, 2), ("2026-08-01", "2026-08-04", 1)],
)
def test_travel_explore_infers_type_from_dates(
    outbound_date: str | None,
    return_date: str | None,
    expected_type: int | None,
) -> None:
    client = FakeClient()
    tool = travel_explore_search(
        provider="function",
        client=client,
        default_params={"arrival_id": "LAX", "type": 1},
    )

    tool(
        departure_id="JFK",
        arrival_area_id="/m/02j9z",
        outbound_date=outbound_date,
        return_date=return_date,
    )

    request = client.calls[0]
    assert request["engine"] == "google_travel_explore"
    assert request["departure_id"] == "JFK"
    assert request["arrival_area_id"] == "/m/02j9z"
    assert "q" not in request
    if expected_type is None:
        assert "type" not in request
    else:
        assert request["type"] == expected_type


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (
            hotels_search,
            {"query": "", "check_in_date": "2026-08-01", "check_out_date": "2026-08-04"},
            "query must not be empty",
        ),
        (
            hotels_search,
            {
                "query": "Kyoto",
                "check_in_date": "2026/08/01",
                "check_out_date": "2026-08-04",
            },
            "check_in_date must be an ISO date",
        ),
        (
            hotels_search,
            {
                "query": "Kyoto",
                "check_in_date": "2026-08-04",
                "check_out_date": "2026-08-01",
            },
            "check_out_date must be after check_in_date",
        ),
        (
            hotels_search,
            {
                "query": "Kyoto",
                "check_in_date": "2026-08-01",
                "check_out_date": "2026-08-04",
                "children": 2,
                "children_ages": [8],
            },
            "children_ages must contain one age per child",
        ),
        (
            flights_search,
            {"departure_id": "", "arrival_id": "AUS", "outbound_date": "2026-08-01"},
            "departure_id must not be empty",
        ),
        (
            flights_search,
            {
                "departure_id": "LAX",
                "arrival_id": "AUS",
                "outbound_date": "2026-08-04",
                "return_date": "2026-08-01",
            },
            "return_date must not be before outbound_date",
        ),
        (
            flights_search,
            {
                "departure_id": "LAX",
                "arrival_id": "AUS",
                "outbound_date": "2026-08-01",
                "adults": 1,
                "infants_on_lap": 2,
            },
            "infants_on_lap cannot exceed adults",
        ),
        (
            travel_explore_search,
            {"departure_id": "JFK", "return_date": "2026-08-04"},
            "return_date requires outbound_date",
        ),
        (
            travel_explore_search,
            {"departure_id": "JFK", "adults": 1, "infants_on_lap": 2},
            "infants_on_lap cannot exceed adults",
        ),
        (
            travel_explore_search,
            {
                "departure_id": "JFK",
                "arrival_id": "LAX",
                "arrival_area_id": "/m/0r4w",
            },
            "arrival_id and arrival_area_id cannot be combined",
        ),
    ],
)
def test_invalid_travel_inputs_fail_before_client_call(factory, kwargs, message: str) -> None:
    tool = factory(provider="function", client=FailIfCalledClient())

    with pytest.raises(ValueError, match=message):
        tool(**kwargs)


@pytest.mark.parametrize("invalid_date", ["20260801", "2026-W31-6"])
@pytest.mark.parametrize(
    ("factory", "kwargs", "date_field"),
    [
        (
            hotels_search,
            {"query": "Kyoto", "check_out_date": "2026-08-04"},
            "check_in_date",
        ),
        (
            flights_search,
            {"departure_id": "LAX", "arrival_id": "AUS"},
            "outbound_date",
        ),
        (travel_explore_search, {"departure_id": "JFK"}, "outbound_date"),
    ],
)
def test_travel_tools_reject_noncanonical_iso_dates(
    factory,
    kwargs: dict[str, object],
    date_field: str,
    invalid_date: str,
) -> None:
    tool = factory(provider="function", client=FailIfCalledClient())

    with pytest.raises(ValueError, match=rf"{date_field} must be an ISO date"):
        tool(**kwargs, **{date_field: invalid_date})


@pytest.mark.parametrize(
    ("factory", "default_params", "valid_call", "message"),
    [
        (
            flights_search,
            {"include_airlines": "AA", "exclude_airlines": "UA"},
            {"departure_id": "LAX", "arrival_id": "AUS", "outbound_date": "2026-08-01"},
            "include_airlines and exclude_airlines",
        ),
        (
            flights_search,
            {"return_times": "8,12"},
            {"departure_id": "LAX", "arrival_id": "AUS", "outbound_date": "2026-08-01"},
            "return_times is supported only for round trips",
        ),
        (
            flights_search,
            {"departure_token": "departure", "booking_token": "booking"},
            {"departure_id": "LAX", "arrival_id": "AUS", "outbound_date": "2026-08-01"},
            "departure_token and booking_token",
        ),
        (
            flights_search,
            {"exclude_basic": True},
            {"departure_id": "LAX", "arrival_id": "AUS", "outbound_date": "2026-08-01"},
            "exclude_basic requires gl='us'",
        ),
        (
            flights_search,
            {"exclude_basic": True, "gl": "us"},
            {
                "departure_id": "LAX",
                "arrival_id": "AUS",
                "outbound_date": "2026-08-01",
                "travel_class": TravelClass.BUSINESS,
            },
            "exclude_basic requires economy",
        ),
        (
            flights_search,
            {"bags": 3},
            {
                "departure_id": "LAX",
                "arrival_id": "AUS",
                "outbound_date": "2026-08-01",
                "adults": 1,
                "children": 1,
            },
            "bags cannot exceed",
        ),
        (
            travel_explore_search,
            {"travel_mode": "1", "interest": "beaches"},
            {"departure_id": "JFK"},
            "travel_mode and interest",
        ),
    ],
)
def test_documented_travel_filter_conflicts_fail_before_client_call(
    factory,
    default_params: dict[str, object],
    valid_call: dict[str, object],
    message: str,
) -> None:
    tool = factory(
        provider="function",
        client=FailIfCalledClient(),
        default_params=default_params,
    )

    with pytest.raises(ValueError, match=message):
        tool(**valid_call)
