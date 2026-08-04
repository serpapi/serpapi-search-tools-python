from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from serpapi_search_tools._adapters import as_provider_tool
from serpapi_search_tools._shared import (
    ProviderName,
    SearchClient,
    SearchResultMode,
    SearchRuntime,
    ToolDefinition,
    object_schema,
    parse_iso_date,
    require_nonempty,
    require_nonnegative,
    require_positive,
)


class TravelClass(str, Enum):
    """Cabin class shared by flight and travel-explore tools."""

    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


TRAVEL_CLASS_TO_SERPAPI: Mapping[TravelClass, int] = {
    TravelClass.ECONOMY: 1,
    TravelClass.PREMIUM_ECONOMY: 2,
    TravelClass.BUSINESS: 3,
    TravelClass.FIRST: 4,
}


def _runtime(
    *,
    api_key: str | None,
    client: SearchClient | None,
    default_params: Mapping[str, Any] | None,
    timeout: float | None,
    mode: SearchResultMode | str,
) -> SearchRuntime:
    return SearchRuntime(
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
    )


def _travel_class_value(value: TravelClass | str) -> int:
    try:
        normalized = TravelClass(value)
    except ValueError as exc:
        choices = ", ".join(member.value for member in TravelClass)
        raise ValueError(f"travel_class must be one of: {choices}.") from exc
    return TRAVEL_CLASS_TO_SERPAPI[normalized]


def _normalize_airport_codes(value: str) -> str:
    """Uppercase IATA airport codes while preserving SerpApi location IDs."""

    segments = (segment.strip() for segment in value.split(","))
    return ",".join(
        segment.upper() if len(segment) == 3 and segment.isalpha() else segment
        for segment in segments
    )


def _validate_passengers(
    *,
    adults: int,
    children: int,
    infants_in_seat: int = 0,
    infants_on_lap: int = 0,
) -> None:
    require_positive(adults, field="adults")
    require_nonnegative(children, field="children")
    require_nonnegative(infants_in_seat, field="infants_in_seat")
    require_nonnegative(infants_on_lap, field="infants_on_lap")
    if infants_on_lap > adults:
        raise ValueError("infants_on_lap cannot exceed adults; each lap infant needs an adult.")


def _validate_flight_filters(params: Mapping[str, Any]) -> None:
    if "q" in params:
        raise ValueError("Flight and travel-explore tools do not support a q parameter.")
    if {"include_airlines", "exclude_airlines"}.issubset(params):
        raise ValueError("include_airlines and exclude_airlines cannot be combined.")
    if {"departure_token", "booking_token"}.issubset(params):
        raise ValueError("departure_token and booking_token cannot be combined.")
    if "return_times" in params and params.get("type") != 1:
        raise ValueError("return_times is supported only for round trips.")
    if params.get("exclude_basic"):
        if params.get("gl") != "us":
            raise ValueError("exclude_basic requires gl='us'.")
        if params.get("travel_class") != 1:
            raise ValueError("exclude_basic requires economy travel_class.")
    if "bags" in params:
        bags = params["bags"]
        if isinstance(bags, bool) or not isinstance(bags, int) or bags < 0:
            raise ValueError("bags must be a nonnegative integer.")
        eligible_passengers = sum(
            value
            for key in ("adults", "children", "infants_in_seat")
            if isinstance((value := params.get(key)), int) and not isinstance(value, bool)
        )
        if bags > eligible_passengers:
            raise ValueError("bags cannot exceed adults + children + infants_in_seat.")


def _common_travel_properties() -> dict[str, dict[str, Any]]:
    return {
        "travel_class": {
            "type": "string",
            "enum": [member.value for member in TravelClass],
            "default": TravelClass.ECONOMY.value,
            "description": "Cabin class.",
        },
        "adults": {
            "type": "integer",
            "minimum": 1,
            "default": 1,
            "description": "Number of adult passengers.",
        },
        "children": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": "Number of child passengers.",
        },
        "infants_in_seat": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": (
                "Infants in their own seats. If seating is unspecified, ask seat or lap."
            ),
        },
        "infants_on_lap": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": (
                "Lap infants, maximum one per adult. If seating is unspecified, ask seat or lap."
            ),
        },
    }


def hotels_search(
    *,
    provider: ProviderName | str = "auto",
    include_examples: bool = True,
    api_key: str | None = None,
    client: SearchClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    name: str = "hotels_search",
) -> Any:
    """Create a Google Hotels tool with explicit stay dates.

    The returned tool requires ``query``, ``check_in_date``, and
    ``check_out_date``. It also exposes typed adult, child, and child-age fields.

    Parameters
    ----------
    provider
        Agent SDK adapter name, ``"auto"``, or ``"function"``.
    include_examples
        Add a short call example to the model-facing description.
    api_key, client, default_params, timeout, mode
        Runtime authentication, custom client, application filters, timeout, and
        compact or full response selection.
    name
        Tool name presented to the model.

    Returns
    -------
    Any
        A provider-native Google Hotels tool with a closed input schema.
    """

    runtime = _runtime(
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
    )

    def hotels_tool(
        query: str,
        check_in_date: str,
        check_out_date: str,
        adults: int = 2,
        children: int = 0,
        children_ages: list[int] | None = None,
    ) -> str:
        require_nonempty({"q": query}, "q", label="query")
        check_in = parse_iso_date(check_in_date, field="check_in_date")
        check_out = parse_iso_date(check_out_date, field="check_out_date")
        if check_out <= check_in:
            raise ValueError("check_out_date must be after check_in_date.")
        require_positive(adults, field="adults")
        require_nonnegative(children, field="children")

        ages = children_ages or []
        if len(ages) != children:
            raise ValueError("children_ages must contain one age per child.")
        for age in ages:
            if isinstance(age, bool) or not isinstance(age, int) or not 1 <= age <= 17:
                raise ValueError("Every child age must be an integer from 1 to 17.")

        typed_params: dict[str, Any] = {
            "q": query,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "children": children,
            "children_ages": None,
        }
        if ages:
            typed_params["children_ages"] = ",".join(str(age) for age in ages)
        return runtime.execute(
            engine="google_hotels",
            typed_params=typed_params,
        )

    description = (
        "Search Google Hotels for properties and rates using explicit check-in and check-out dates."
    )
    if include_examples:
        description += " Example: search for hotels in Kyoto using explicit future dates."
    schema = object_schema(
        {
            "query": {"type": "string", "description": "Hotel or destination query."},
            "check_in_date": {
                "type": "string",
                "format": "date",
                "description": "Future check-in date in YYYY-MM-DD format.",
            },
            "check_out_date": {
                "type": "string",
                "format": "date",
                "description": (
                    "Check-out date in YYYY-MM-DD format; must be after check_in_date."
                ),
            },
            "adults": {
                "type": "integer",
                "minimum": 1,
                "default": 2,
                "description": "Number of adult guests.",
            },
            "children": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "Number of child guests.",
            },
            "children_ages": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 17},
                "description": (
                    "One age (1-17) per child; use 1 for infants under one. Omit when children=0."
                ),
            },
        },
        required=["query", "check_in_date", "check_out_date"],
    )
    hotels_tool.__annotations__ = {
        "query": str,
        "check_in_date": str,
        "check_out_date": str,
        "adults": int,
        "children": int,
        "children_ages": list[int] | None,
        "return": str,
    }
    return as_provider_tool(ToolDefinition(hotels_tool, name, description, schema), provider)


def flights_search(
    *,
    provider: ProviderName | str = "auto",
    include_examples: bool = True,
    api_key: str | None = None,
    client: SearchClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    name: str = "flights_search",
) -> Any:
    """Create a Google Flights tool for one-way and round-trip routes.

    The returned tool requires departure, arrival, and outbound date fields.
    Adding ``return_date`` selects a round trip. Cabin class and passenger
    counts are semantic typed fields; no synthetic ``q`` field is exposed.

    Parameters
    ----------
    provider
        Agent SDK adapter name, ``"auto"``, or ``"function"``.
    include_examples
        Add a short call example to the model-facing description.
    api_key, client, default_params, timeout, mode
        Runtime authentication, custom client, application filters, timeout, and
        compact or full response selection.
    name
        Tool name presented to the model.

    Returns
    -------
    Any
        A provider-native Google Flights route-search tool.
    """

    runtime = _runtime(
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
    )

    def flights_tool(
        departure_id: str,
        arrival_id: str,
        outbound_date: str,
        return_date: str | None = None,
        travel_class: TravelClass = TravelClass.ECONOMY,
        adults: int = 1,
        children: int = 0,
        infants_in_seat: int = 0,
        infants_on_lap: int = 0,
    ) -> str:
        departure = _normalize_airport_codes(
            require_nonempty({"departure_id": departure_id}, "departure_id")
        )
        arrival = _normalize_airport_codes(
            require_nonempty({"arrival_id": arrival_id}, "arrival_id")
        )
        outbound = parse_iso_date(outbound_date, field="outbound_date")
        if return_date is not None:
            returning = parse_iso_date(return_date, field="return_date")
            if returning < outbound:
                raise ValueError("return_date must not be before outbound_date.")
        _validate_passengers(
            adults=adults,
            children=children,
            infants_in_seat=infants_in_seat,
            infants_on_lap=infants_on_lap,
        )
        typed_params = {
            "departure_id": departure,
            "arrival_id": arrival,
            "outbound_date": outbound_date,
            "return_date": return_date,
            "type": 1 if return_date is not None else 2,
            "travel_class": _travel_class_value(travel_class),
            "adults": adults,
            "children": children,
            "infants_in_seat": infants_in_seat,
            "infants_on_lap": infants_on_lap,
        }

        def validate(params: dict[str, Any]) -> None:
            _validate_flight_filters(params)

        return runtime.execute(
            engine="google_flights",
            typed_params=typed_params,
            validator=validate,
        )

    description = (
        "Search Google Flights for a known one-way or round-trip route using airport IATA "
        "codes or city KGMIDs (Freebase IDs) and explicit future travel dates. Only use a "
        "city ID when the exact ID is supplied; for a city name, ask for a specific airport. "
        "Never infer an ID or use a metropolitan code."
    )
    if include_examples:
        description += (
            " Example format: use an airport's three-letter IATA code for a specific airport, "
            "or a city's /m/ or /g/ KGMID/Freebase ID for city-wide results."
        )
    properties = {
        "departure_id": {
            "type": "string",
            "description": (
                "Specific departure airport IATA code or exact user-supplied city KGMID "
                "(Freebase ID, /m/ or /g/); never infer IDs or use metropolitan codes; "
                "comma-separate multiple values."
            ),
        },
        "arrival_id": {
            "type": "string",
            "description": (
                "Specific arrival airport IATA code or exact user-supplied city KGMID "
                "(Freebase ID, /m/ or /g/); never infer IDs or use metropolitan codes; "
                "comma-separate multiple values."
            ),
        },
        "outbound_date": {
            "type": "string",
            "format": "date",
            "description": "Future outbound date in YYYY-MM-DD format.",
        },
        "return_date": {
            "type": "string",
            "format": "date",
            "description": (
                "Return date for round trips; omit for one-way. Must be on or after outbound_date."
            ),
        },
        **_common_travel_properties(),
    }
    schema = object_schema(
        properties,
        required=["departure_id", "arrival_id", "outbound_date"],
    )
    flights_tool.__annotations__ = {
        "departure_id": str,
        "arrival_id": str,
        "outbound_date": str,
        "return_date": str | None,
        "travel_class": TravelClass,
        "adults": int,
        "children": int,
        "infants_in_seat": int,
        "infants_on_lap": int,
        "return": str,
    }
    return as_provider_tool(ToolDefinition(flights_tool, name, description, schema), provider)


def travel_explore_search(
    *,
    provider: ProviderName | str = "auto",
    include_examples: bool = True,
    api_key: str | None = None,
    client: SearchClient | None = None,
    default_params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    mode: SearchResultMode | str = SearchResultMode.COMPACT,
    name: str = "travel_explore_search",
) -> Any:
    """Create a Google Travel Explore destination-discovery tool.

    The returned tool requires only ``departure_id`` and offers optional
    destination, date, cabin-class, and passenger fields. It never exposes or
    sends a synthetic ``q`` parameter.

    Parameters
    ----------
    provider
        Agent SDK adapter name, ``"auto"``, or ``"function"``.
    include_examples
        Add a short call example to the model-facing description.
    api_key, client, default_params, timeout, mode
        Runtime authentication, custom client, application filters, timeout, and
        compact or full response selection.
    name
        Tool name presented to the model.

    Returns
    -------
    Any
        A provider-native Google Travel Explore tool.
    """

    runtime = _runtime(
        api_key=api_key,
        client=client,
        default_params=default_params,
        timeout=timeout,
        mode=mode,
    )

    def travel_explore_tool(
        departure_id: str,
        arrival_id: str | None = None,
        arrival_area_id: str | None = None,
        outbound_date: str | None = None,
        return_date: str | None = None,
        travel_class: TravelClass = TravelClass.ECONOMY,
        adults: int = 1,
        children: int = 0,
        infants_in_seat: int = 0,
        infants_on_lap: int = 0,
    ) -> str:
        departure = _normalize_airport_codes(
            require_nonempty({"departure_id": departure_id}, "departure_id")
        )
        if arrival_id is not None:
            arrival_id = _normalize_airport_codes(
                require_nonempty({"arrival_id": arrival_id}, "arrival_id")
            )
        if arrival_area_id is not None:
            arrival_area_id = require_nonempty(
                {"arrival_area_id": arrival_area_id}, "arrival_area_id"
            )
        if arrival_id is not None and arrival_area_id is not None:
            raise ValueError("arrival_id and arrival_area_id cannot be combined.")
        if return_date is not None and outbound_date is None:
            raise ValueError("return_date requires outbound_date.")
        outbound = None
        if outbound_date is not None:
            outbound = parse_iso_date(outbound_date, field="outbound_date")
        if return_date is not None:
            returning = parse_iso_date(return_date, field="return_date")
            if outbound is not None and returning < outbound:
                raise ValueError("return_date must not be before outbound_date.")
        _validate_passengers(
            adults=adults,
            children=children,
            infants_in_seat=infants_in_seat,
            infants_on_lap=infants_on_lap,
        )
        typed_params: dict[str, Any] = {
            "departure_id": departure,
            "arrival_id": arrival_id,
            "arrival_area_id": arrival_area_id,
            "outbound_date": outbound_date,
            "return_date": return_date,
            "type": None,
            "travel_class": _travel_class_value(travel_class),
            "adults": adults,
            "children": children,
            "infants_in_seat": infants_in_seat,
            "infants_on_lap": infants_on_lap,
        }
        if outbound_date is not None:
            typed_params["type"] = 1 if return_date is not None else 2

        def validate(params: dict[str, Any]) -> None:
            _validate_flight_filters(params)
            if {"travel_mode", "interest"}.issubset(params):
                raise ValueError(
                    "Google Travel Explore travel_mode and interest cannot be combined."
                )

        return runtime.execute(
            engine="google_travel_explore",
            typed_params=typed_params,
            validator=validate,
        )

    description = (
        "Explore destinations and fares from a departure airport or city with optional "
        "specific-destination, regional, and date constraints."
    )
    if include_examples:
        description += " Example: departure_id='JFK', arrival_area_id='/m/02j9z' for Europe."
    properties = {
        "departure_id": {
            "type": "string",
            "description": (
                "Departure airport IATA code or city KGMID (/m/ or /g/); comma-separate "
                "multiple values."
            ),
        },
        "arrival_id": {
            "type": "string",
            "description": (
                "Arrival airport IATA code or city KGMID (/m/ or /g/). Use arrival_area_id "
                "for regions; the fields are mutually exclusive."
            ),
        },
        "arrival_area_id": {
            "type": "string",
            "description": (
                "Region or country KGMID (/m/ or /g/); mutually exclusive with arrival_id."
            ),
        },
        "outbound_date": {
            "type": "string",
            "format": "date",
            "description": "Future outbound date; omit for flexible dates.",
        },
        "return_date": {
            "type": "string",
            "format": "date",
            "description": (
                "Return date; requires outbound_date. Omit for one-way or flexible dates."
            ),
        },
        **_common_travel_properties(),
    }
    schema = object_schema(properties, required=["departure_id"])
    travel_explore_tool.__annotations__ = {
        "departure_id": str,
        "arrival_id": str | None,
        "arrival_area_id": str | None,
        "outbound_date": str | None,
        "return_date": str | None,
        "travel_class": TravelClass,
        "adults": int,
        "children": int,
        "infants_in_seat": int,
        "infants_on_lap": int,
        "return": str,
    }
    return as_provider_tool(
        ToolDefinition(travel_explore_tool, name, description, schema), provider
    )
