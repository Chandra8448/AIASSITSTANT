"""CrewAI-based travel planning demo with sourced research and mock bookings."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any
from urllib.parse import quote

import requests
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from litellm.exceptions import APIError
from openai import APIError as OpenAIAPIError

REQUEST_TIMEOUT_SECONDS = 12
USER_AGENT = "ClaudePracticeTravelPlanner/1.0 (training demo)"
FORECAST_DAYS = 16


class TravelPlannerError(ValueError):
    """Raised when travel input is invalid or a required service is unavailable."""


@dataclass(frozen=True)
class TravelRequest:
    destination: str
    starting_location: str
    start_date: date
    end_date: date
    budget_inr: float
    interests: tuple[str, ...]

    @property
    def duration_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def nights(self) -> int:
        return max(0, (self.end_date - self.start_date).days)


def validate_travel_request(request: TravelRequest, today: date | None = None) -> None:
    """Validate the customer-provided values before making API calls."""
    today = today or date.today()
    if not request.destination.strip():
        raise TravelPlannerError("Enter a destination.")
    if not request.starting_location.strip():
        raise TravelPlannerError("Enter a starting location.")
    if request.start_date < today:
        raise TravelPlannerError("The travel start date cannot be in the past.")
    if request.end_date < request.start_date:
        raise TravelPlannerError("The return date must be on or after the start date.")
    if request.duration_days > 30:
        raise TravelPlannerError("This demo supports trips up to 30 days.")
    if request.budget_inr <= 0:
        raise TravelPlannerError("The total budget must be greater than zero.")
    if not request.interests:
        raise TravelPlannerError("Choose at least one travel interest.")


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise TravelPlannerError(f"External travel data could not be retrieved: {exc}") from exc
    if not isinstance(payload, dict):
        raise TravelPlannerError("An external travel service returned an unexpected response.")
    return payload


def _geocode(place: str) -> dict[str, Any]:
    payload = _get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": place, "count": 1, "language": "en", "format": "json"},
    )
    results = payload.get("results", [])
    if not results:
        raise TravelPlannerError(f"Could not find coordinates for {place!r}.")
    try:
        place_data = results[0]
        return {
            "name": place_data.get("name", place),
            "latitude": float(place_data["latitude"]),
            "longitude": float(place_data["longitude"]),
            "country": place_data.get("country", ""),
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise TravelPlannerError("The geocoding service returned incomplete coordinates.") from exc


def search_destination(destination: str, interests: tuple[str, ...]) -> dict[str, Any]:
    """Search Tavily when configured, otherwise use the public Wikipedia API."""
    query = f"{destination} attractions local experiences {', '.join(interests)}"
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                },
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise TravelPlannerError(f"Tavily destination search failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise TravelPlannerError("Tavily returned an unexpected search response.")
        results = [
            {
                "title": item.get("title", ""),
                "summary": item.get("content", ""),
                "url": item.get("url", ""),
            }
            for item in payload.get("results", [])
        ]
        return {"provider": "Tavily Search API", "query": query, "results": results}

    payload = _get_json(
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 5,
        },
    )
    results = [
        {
            "title": item.get("title", ""),
            "summary": item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
            "url": f"https://en.wikipedia.org/wiki/{quote(item.get('title', '').replace(' ', '_'))}",
        }
        for item in payload.get("query", {}).get("search", [])
    ]
    return {"provider": "Wikipedia public search API", "query": query, "results": results}


def get_weather_forecast(
    destination: str, start_date: date, end_date: date, today: date | None = None
) -> dict[str, Any]:
    """Fetch an Open-Meteo forecast when the requested dates are in range."""
    today = today or date.today()
    if (start_date - today).days > FORECAST_DAYS:
        return {
            "available": False,
            "message": f"A reliable daily forecast is available only within {FORECAST_DAYS} days of travel.",
            "source": "https://open-meteo.com/",
        }
    if (end_date - today).days > FORECAST_DAYS:
        return {
            "available": False,
            "message": f"The full trip is outside the {FORECAST_DAYS}-day forecast window; no partial forecast shown.",
            "source": "https://open-meteo.com/",
        }

    location = _geocode(destination)
    payload = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "timezone": "auto",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    daily = payload.get("daily", {})
    if not isinstance(daily, dict) or not daily.get("time"):
        return {
            "available": False,
            "message": "The weather service returned no daily forecast for these dates.",
            "location": location,
            "source": "https://open-meteo.com/",
        }
    return {
        "available": True,
        "location": location,
        "daily": daily,
        "source": "https://open-meteo.com/",
    }


def get_route_estimate(starting_location: str, destination: str) -> dict[str, Any]:
    """Estimate road distance/time using Open-Meteo geocoding and public OSRM."""
    origin = _geocode(starting_location)
    target = _geocode(destination)
    coordinates = (
        f"{origin['longitude']},{origin['latitude']};"
        f"{target['longitude']},{target['latitude']}"
    )
    payload = _get_json(f"https://router.project-osrm.org/route/v1/driving/{coordinates}", {"overview": "false"})
    routes = payload.get("routes", [])
    if not routes:
        raise TravelPlannerError("The routing service did not find a road route for those locations.")
    try:
        route = routes[0]
        return {
            "distance_km": round(float(route["distance"]) / 1000, 1),
            "driving_hours": round(float(route["duration"]) / 3600, 1),
            "origin": origin,
            "destination": target,
            "source": "https://project-osrm.org/",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise TravelPlannerError("The routing service returned incomplete distance data.") from exc


def get_mock_booking_options(
    destination: str, nights: int, distance_km: float | None
) -> dict[str, Any]:
    """Return illustrative options; this does not query inventory or place bookings."""
    estimated_distance = max(0.0, distance_km or 0.0)
    transport = max(1200, round(estimated_distance * 5))
    return {
        "notice": (
            "Mock estimates only. No live hotel/flight inventory or booking API is connected. "
            "Intercity transport uses a rough distance-based formula; when routing is unavailable, "
            "a ₹1,200 placeholder is used."
        ),
        "transport_estimate_basis": (
            f"5 INR per road-km, minimum ₹1,200; road distance {estimated_distance:g} km."
            if distance_km is not None
            else "Routing unavailable; ₹1,200 placeholder, not a distance-based fare."
        ),
        "hotels": [
            {"name": "Value stay (mock)", "price_per_night_inr": 1800, "nights": nights},
            {"name": "Comfort stay (mock)", "price_per_night_inr": 3500, "nights": nights},
            {"name": "Premium stay (mock)", "price_per_night_inr": 6000, "nights": nights},
        ],
        "transport": [
            {
                "name": "Budget intercity transport (mock)",
                "round_trip_estimate_inr": max(700, round(estimated_distance * 2)),
            },
            {
                "name": "Standard intercity transport (mock)",
                "round_trip_estimate_inr": transport,
            },
        ],
        "local_transport_per_day_inr": 450,
        "food_per_day_inr": 800,
        "activities_per_day_inr": 400,
        "destination": destination,
    }


def calculate_expenses(
    total_budget_inr: float,
    hotel_per_night_inr: float,
    nights: int,
    intercity_transport_inr: float,
    daily_local_transport_inr: float,
    daily_food_inr: float,
    daily_activities_inr: float,
    contingency_percent: float = 10,
    duration_days: int | None = None,
) -> dict[str, Any]:
    """Calculate the authoritative prototype cost estimate in Python."""
    values = (
        total_budget_inr,
        hotel_per_night_inr,
        intercity_transport_inr,
        daily_local_transport_inr,
        daily_food_inr,
        daily_activities_inr,
    )
    if any(not math.isfinite(value) or value < 0 for value in values) or nights < 0:
        raise TravelPlannerError("Expense estimates must be non-negative.")
    if not math.isfinite(contingency_percent) or not 0 <= contingency_percent <= 100:
        raise TravelPlannerError("Contingency must be between 0 and 100 percent.")
    days = duration_days or nights + 1
    breakdown = {
        "hotel": round(hotel_per_night_inr * nights, 2),
        "intercity_transport": round(intercity_transport_inr, 2),
        "local_transport": round(daily_local_transport_inr * days, 2),
        "food": round(daily_food_inr * days, 2),
        "activities": round(daily_activities_inr * days, 2),
    }
    subtotal = round(sum(breakdown.values()), 2)
    contingency = round(subtotal * contingency_percent / 100, 2)
    total = round(subtotal + contingency, 2)
    return {
        "currency": "INR",
        "breakdown_inr": breakdown,
        "subtotal_inr": subtotal,
        "contingency_percent": contingency_percent,
        "contingency_inr": contingency,
        "total_estimate_inr": total,
        "budget_inr": round(total_budget_inr, 2),
        "remaining_inr": round(total_budget_inr - total, 2),
        "within_budget": total <= total_budget_inr,
    }


def _calculate_profile(
    options: dict[str, Any], request: TravelRequest, profile: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    food_activity_days = request.duration_days
    hotel = options["hotels"][1] if profile == "standard" else options["hotels"][0]
    transport = options["transport"][1] if profile == "standard" else options["transport"][0]
    expenses = calculate_expenses(
        request.budget_inr,
        hotel["price_per_night_inr"],
        request.nights,
        transport["round_trip_estimate_inr"],
        options["local_transport_per_day_inr"] if profile == "standard" else 300,
        options["food_per_day_inr"] if profile == "standard" else 600,
        options["activities_per_day_inr"] if profile == "standard" else 250,
        duration_days=food_activity_days,
    )
    return {"hotel": hotel, "transport": transport, "profile": profile}, expenses


def _make_cached_tools(
    research: dict[str, Any],
    weather: dict[str, Any],
    route: dict[str, Any],
    options: dict[str, Any],
    request: TravelRequest,
) -> list[Any]:
    """Expose API results and verified calculations as callable CrewAI tools."""

    @tool("destination_search")
    def destination_search(query: str) -> str:
        """Return sourced search results for destination attractions and experiences."""
        del query
        return json.dumps(research, ensure_ascii=False)

    @tool("destination_weather")
    def destination_weather(location: str, travel_dates: str) -> str:
        """Return an Open-Meteo forecast for the requested destination and dates, if available."""
        del location, travel_dates
        return json.dumps(weather, ensure_ascii=False)

    @tool("route_estimator")
    def route_estimator(origin: str, destination_name: str) -> str:
        """Return a public OSRM road distance and driving-time estimate."""
        del origin, destination_name
        return json.dumps(route, ensure_ascii=False)

    @tool("mock_hotel_transport_options")
    def mock_hotel_transport_options(destination_name: str, trip_nights: int) -> str:
        """Return clearly labeled mock hotel and transport options; no booking is made."""
        del destination_name, trip_nights
        return json.dumps(options, ensure_ascii=False)

    @tool("travel_budget_calculator")
    def travel_budget_calculator(
        total_budget_inr: float,
        hotel_per_night_inr: float,
        nights: int,
        intercity_transport_inr: float,
        daily_local_transport_inr: float,
        daily_food_inr: float,
        daily_activities_inr: float,
        contingency_percent: float = 10,
    ) -> str:
        """Calculate trip costs in INR, including contingency and budget headroom."""
        return json.dumps(
            calculate_expenses(
                total_budget_inr,
                hotel_per_night_inr,
                nights,
                intercity_transport_inr,
                daily_local_transport_inr,
                daily_food_inr,
                daily_activities_inr,
                contingency_percent,
                request.duration_days,
            ),
            ensure_ascii=False,
        )

    @tool("daily_itinerary_scheduler")
    def daily_itinerary_scheduler(start_day: str, number_of_days: int, interest_list: str) -> str:
        """Create dated day-by-day planning slots based on customer interests."""
        day_zero = date.fromisoformat(start_day)
        interests = [interest.strip() for interest in interest_list.split(",") if interest.strip()]
        if not interests:
            interests = ["Sightseeing"]
        slots = [
            {
                "day": index + 1,
                "date": (day_zero + timedelta(days=index)).isoformat(),
                "morning": f"One researched {interests[index % len(interests)].lower()} experience",
                "afternoon": f"Explore another {interests[(index + 1) % len(interests)].lower()} option",
                "evening": "Flexible time for dining or rest",
            }
            for index in range(number_of_days)
        ]
        return json.dumps(slots, ensure_ascii=False)

    return [
        destination_search,
        destination_weather,
        route_estimator,
        mock_hotel_transport_options,
        travel_budget_calculator,
        daily_itinerary_scheduler,
    ]


def _create_llm(provider: str, model: str | None = None) -> LLM:
    provider = provider.strip().lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise TravelPlannerError("Set OPENAI_API_KEY in the environment to use OpenAI.")
        selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return LLM(
            model=selected_model,
            api_key=api_key,
            temperature=0.2,
        )
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise TravelPlannerError("Set OPENROUTER_API_KEY in the environment to use OpenRouter.")
        selected_model = model or os.getenv(
            "OPENROUTER_MODEL", "openrouter/openai/gpt-4o-mini"
        )
        return LLM(
            model=selected_model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
        )
    if provider == "ollama":
        selected_model = model or os.getenv("OLLAMA_MODEL", "ollama/llama3.1")
        return LLM(
            model=selected_model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.2,
        )
    raise TravelPlannerError("Choose OpenAI, OpenRouter, or Ollama as the LLM provider.")


def _build_crew(
    request: TravelRequest,
    provider: str,
    model: str | None,
    tools: list[Any],
    research: dict[str, Any],
    weather: dict[str, Any],
    route: dict[str, Any],
    options: dict[str, Any],
    expenses: dict[str, Any],
    revision_context: str = "",
) -> Crew:
    llm = _create_llm(provider, model)
    coordinator = Agent(
        role="Travel Coordinator",
        goal="Coordinate the specialists, reconcile their evidence, and deliver a safe and coherent customer plan.",
        backstory="You are a careful travel operations lead. You distinguish verified facts from estimates and never claim a booking was made.",
        llm=llm,
        verbose=False,
    )
    destination_agent = Agent(
        role="Destination Research Agent",
        goal="Find relevant attractions and local experiences, using sourced search and weather data.",
        backstory="You are a destination researcher who cites sources and marks unavailable or uncertain information.",
        tools=[tools[0], tools[1]],
        llm=llm,
        verbose=False,
    )
    hotel_transport_agent = Agent(
        role="Hotel and Transport Agent",
        goal="Use route and mock options to recommend practical accommodation and transport within the trip constraints.",
        backstory="You never present mock prices as live availability and you do not claim to book travel.",
        tools=[tools[2], tools[3]],
        llm=llm,
        verbose=False,
    )
    budget_agent = Agent(
        role="Budget Analyst",
        goal="Use the calculator tool to verify the full trip estimate and identify any budget gap.",
        backstory="You include contingency, use only supplied estimates, and clearly distinguish estimates from quotes.",
        tools=[tools[4]],
        llm=llm,
        verbose=False,
    )
    itinerary_agent = Agent(
        role="Itinerary Planner",
        goal="Create a feasible date-by-date itinerary that respects the selected activities, route, weather, and cost constraints.",
        backstory="You make practical travel schedules with room for meals, transit, and rest.",
        tools=[tools[5]],
        llm=llm,
        verbose=False,
    )

    trip = asdict(request)
    trip["start_date"] = request.start_date.isoformat()
    trip["end_date"] = request.end_date.isoformat()
    trip["duration_days"] = request.duration_days
    trip["nights"] = request.nights
    trip_text = json.dumps(trip, ensure_ascii=False)
    selected = options["selected"]
    budget_tool_inputs = {
        "total_budget_inr": request.budget_inr,
        "hotel_per_night_inr": selected["hotel"]["price_per_night_inr"],
        "nights": request.nights,
        "intercity_transport_inr": selected["transport"]["round_trip_estimate_inr"],
        "daily_local_transport_inr": options["selected_daily_local_transport_inr"],
        "daily_food_inr": options["selected_daily_food_inr"],
        "daily_activities_inr": options["selected_daily_activities_inr"],
        "contingency_percent": 10,
    }
    task_research = Task(
        description=(
            "Research the customer's destination and interests. Call destination_search and destination_weather. "
            "Summarize relevant findings and cite returned URLs. If a tool reports an error or forecast unavailable, "
            "say so explicitly; do not invent attractions, forecasts, opening hours, or citations.\n\n"
            f"Customer request: {trip_text}"
        ),
        expected_output="Destination findings, date-relevant weather if available, and source URLs; uncertainties are explicit.",
        agent=destination_agent,
    )
    task_options = Task(
        description=(
            "Call route_estimator and mock_hotel_transport_options. Recommend an appropriate option, state that prices "
            "are mock estimates and not live inventory, and use the route data only as an estimate.\n\n"
            f"Customer request: {trip_text}\n"
            f"Candidate options: {json.dumps(options, ensure_ascii=False)}"
        ),
        expected_output="Road distance/time when available and one clearly labeled mock hotel and transport recommendation.",
        agent=hotel_transport_agent,
    )
    task_budget = Task(
        description=(
            "Call travel_budget_calculator with the selected lodging, transport, food, local transport, activity, "
            "duration, and contingency inputs. Use these exact numeric inputs: "
            f"{json.dumps(budget_tool_inputs)}. Compare the tool result to the customer's budget. "
            f"Verified Python calculation: {json.dumps(expenses)}\n"
            f"Selected options: {json.dumps(options, ensure_ascii=False)}\n"
            f"Customer request: {trip_text}"
        ),
        expected_output="Expense breakdown, estimated total, budget comparison, and cheaper recommendations if needed.",
        agent=budget_agent,
        context=[task_options],
    )
    task_itinerary = Task(
        description=(
            "Call daily_itinerary_scheduler and prepare a day-by-day plan based on research, budget, and travel options. "
            "Use only the provided facts. Include practical meal/rest/transit notes and source links. "
            f"Weather: {json.dumps(weather, ensure_ascii=False)}\n"
            f"Route: {json.dumps(route, ensure_ascii=False)}\n"
            f"Budget: {json.dumps(expenses, ensure_ascii=False)}\n"
            f"Revision context: {revision_context or 'Initial plan; no prior attempt.'}\n"
            f"Customer request: {trip_text}"
        ),
        expected_output="A dated itinerary, sourced recommendations, selected options, cost summary, and any budget alternatives.",
        agent=itinerary_agent,
        context=[task_research, task_options, task_budget],
    )
    task_final = Task(
        description=(
            "Consolidate the prior task outputs into a customer-ready travel plan. Include dates and day-by-day "
            "activities, attractions, accommodation and transport estimates, expense breakdown and budget comparison, "
            "alternatives if over budget, source URLs, forecast limitations, and mock-booking disclaimers. "
            "Never state that reservations or purchases were made. "
            f"Verified estimate: {json.dumps(expenses)}\n"
            f"Research evidence: {json.dumps(research, ensure_ascii=False)}\n"
            f"Weather data: {json.dumps(weather, ensure_ascii=False)}\n"
            f"Route estimate: {json.dumps(route, ensure_ascii=False)}\n"
            f"Revision context: {revision_context or 'Initial plan.'}"
        ),
        expected_output="A complete accurate travel plan with source links and a clear budget verdict.",
        agent=coordinator,
        context=[task_research, task_options, task_budget, task_itinerary],
    )
    return Crew(
        agents=[
            coordinator,
            destination_agent,
            hotel_transport_agent,
            budget_agent,
            itinerary_agent,
        ],
        tasks=[task_research, task_options, task_budget, task_itinerary, task_final],
        process=Process.sequential,
        verbose=False,
    )


def _kickoff(crew: Crew) -> str:
    try:
        return str(crew.kickoff())
    except (APIError, OpenAIAPIError) as exc:
        raise TravelPlannerError(
            f"The selected LLM provider rejected or could not complete the request "
            f"({type(exc).__name__}). Check that the key matches the selected provider "
            "(OPENAI_API_KEY for OpenAI, OPENROUTER_API_KEY for OpenRouter), and verify "
            "the model name, network connection, and account balance."
        ) from exc


def generate_travel_plan(
    request: TravelRequest,
    provider: str = "ollama",
    model: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Collect API evidence, coordinate CrewAI specialists, and revise over-budget estimates."""
    validate_travel_request(request, today)
    warnings: list[str] = []
    try:
        research = search_destination(request.destination, request.interests)
    except TravelPlannerError as exc:
        research = {"provider": "unavailable", "results": [], "error": str(exc)}
        warnings.append(str(exc))
    try:
        weather = get_weather_forecast(
            request.destination, request.start_date, request.end_date, today
        )
    except TravelPlannerError as exc:
        weather = {"available": False, "error": str(exc), "source": "https://open-meteo.com/"}
        warnings.append(str(exc))
    try:
        route = get_route_estimate(request.starting_location, request.destination)
        distance_km = route["distance_km"]
    except TravelPlannerError as exc:
        route = {"available": False, "error": str(exc), "source": "https://project-osrm.org/"}
        distance_km = None
        warnings.append(str(exc))

    options = get_mock_booking_options(request.destination, request.nights, distance_km)
    initial_selected, initial_expenses = _calculate_profile(options, request, "standard")
    revision_triggered = not initial_expenses["within_budget"]
    if revision_triggered:
        selected, expenses = _calculate_profile(options, request, "budget")
        profile = "budget"
    else:
        selected, expenses, profile = initial_selected, initial_expenses, "standard"
    initial_options = {
        **options,
        "selected": initial_selected,
        "price_profile": "standard",
        "selected_daily_local_transport_inr": 450,
        "selected_daily_food_inr": 800,
        "selected_daily_activities_inr": 400,
    }
    options["selected"] = selected
    options["price_profile"] = profile
    options["selected_daily_local_transport_inr"] = 300 if profile == "budget" else 450
    options["selected_daily_food_inr"] = 600 if profile == "budget" else 800
    options["selected_daily_activities_inr"] = 250 if profile == "budget" else 400
    tools = _make_cached_tools(research, weather, route, initial_options, request)

    first_result = _kickoff(_build_crew(
        request,
        provider,
        model,
        tools,
        research,
        weather,
        route,
        initial_options,
        initial_expenses,
    ))
    final_result = first_result
    alternatives: list[str] = []
    if revision_triggered:
        alternatives.extend(
            [
                "Selected the lowest-cost mock hotel and intercity transport options.",
                "Applied lower daily food, local transport, and activities assumptions.",
            ]
        )
        revision_context = (
            "The initial standard estimate exceeded the budget. The accommodation/transport agent selected "
            "the budget mock options and the budget analyst recalculated. "
            f"Revised estimate: {json.dumps(expenses)}"
        )
        alternatives.append(
            "Revised estimate is within budget."
            if expenses["within_budget"]
            else "The revised estimate remains over budget."
        )
        revised_tools = _make_cached_tools(research, weather, route, options, request)
        final_result = _kickoff(_build_crew(
            request,
            provider,
            model,
            revised_tools,
            research,
            weather,
            route,
            options,
            expenses,
            revision_context,
        ))
        if not expenses["within_budget"]:
            alternatives.append(
                "Even the lowest-cost mock profile exceeds the budget; increase the budget or shorten the trip."
            )

    sources = []
    for result in research.get("results", []):
        url = result.get("url")
        if url and url not in sources:
            sources.append(url)
    if weather.get("source"):
        sources.append(weather["source"])
    if route.get("source"):
        sources.append(route["source"])

    return {
        "plan": str(final_result),
        "initial_plan": str(first_result) if revision_triggered else None,
        "request": {
            "destination": request.destination,
            "starting_location": request.starting_location,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "duration_days": request.duration_days,
            "nights": request.nights,
            "budget_inr": request.budget_inr,
            "interests": list(request.interests),
        },
        "research": research,
        "weather": weather,
        "route": route,
        "options": options,
        "budget": expenses,
        "initial_budget": initial_expenses,
        "revision_triggered": revision_triggered,
        "alternatives": alternatives,
        "sources": sources,
        "warnings": warnings,
        "llm_provider": provider,
        "llm_model": model or "",
    }
