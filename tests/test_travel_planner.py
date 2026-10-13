from __future__ import annotations

import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from travel_planner import (
    TravelPlannerError,
    TravelRequest,
    _calculate_profile,
    _make_cached_tools,
    calculate_expenses,
    generate_travel_plan,
    get_mock_booking_options,
    get_route_estimate,
    get_weather_forecast,
    search_destination,
    validate_travel_request,
)


class TravelPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.today = date(2030, 1, 1)
        self.request = TravelRequest(
            destination="Kochi",
            starting_location="Bengaluru",
            start_date=self.today + timedelta(days=5),
            end_date=self.today + timedelta(days=8),
            budget_inr=30_000,
            interests=("Sightseeing", "Food"),
        )

    def test_validates_complete_travel_request(self) -> None:
        validate_travel_request(self.request, self.today)
        self.assertEqual(self.request.duration_days, 4)
        self.assertEqual(self.request.nights, 3)

    def test_same_day_trip_has_no_hotel_night(self) -> None:
        day_trip = TravelRequest(
            "Kochi", "Bengaluru", self.request.start_date,
            self.request.start_date, 10_000, ("Sightseeing",)
        )
        validate_travel_request(day_trip, self.today)
        self.assertEqual(day_trip.duration_days, 1)
        self.assertEqual(day_trip.nights, 0)

    def test_rejects_past_dates_and_empty_interests(self) -> None:
        past_request = TravelRequest(
            "Kochi", "Bengaluru", self.today - timedelta(days=1),
            self.today, 10_000, ("Culture",)
        )
        with self.assertRaisesRegex(TravelPlannerError, "past"):
            validate_travel_request(past_request, self.today)
        no_interests = TravelRequest(
            "Kochi", "Bengaluru", self.request.start_date,
            self.request.end_date, 10_000, ()
        )
        with self.assertRaisesRegex(TravelPlannerError, "interest"):
            validate_travel_request(no_interests, self.today)

    def test_rejects_invalid_trip_duration_budget_and_order(self) -> None:
        invalid = TravelRequest(
            "Kochi", "Bengaluru", self.request.end_date,
            self.request.start_date, 0, ("Sightseeing",)
        )
        with self.assertRaisesRegex(TravelPlannerError, "return date"):
            validate_travel_request(invalid, self.today)

    def test_calculates_total_cost_with_contingency(self) -> None:
        estimate = calculate_expenses(
            20_000, 2_000, 3, 1_000, 300, 600, 250,
            duration_days=4,
        )
        self.assertEqual(estimate["subtotal_inr"], 11_600)
        self.assertEqual(estimate["contingency_inr"], 1_160)
        self.assertEqual(estimate["total_estimate_inr"], 12_760)
        self.assertTrue(estimate["within_budget"])

    def test_detects_budget_overrun(self) -> None:
        estimate = calculate_expenses(10_000, 6_000, 3, 12_000, 500, 800, 400)
        self.assertFalse(estimate["within_budget"])
        self.assertLess(estimate["remaining_inr"], 0)

    def test_booking_estimates_are_explicitly_mocked(self) -> None:
        options = get_mock_booking_options("Kochi", 3, 500)
        self.assertIn("Mock estimates only", options["notice"])
        self.assertEqual(options["hotels"][0]["price_per_night_inr"], 1_800)
        self.assertIn("5 INR per road-km", options["transport_estimate_basis"])
        unknown_distance = get_mock_booking_options("Kochi", 3, None)
        self.assertIn("Routing unavailable", unknown_distance["transport_estimate_basis"])

    def test_over_budget_standard_option_is_replaced_by_budget_option(self) -> None:
        request = TravelRequest(
            "Kochi", "Bengaluru", self.request.start_date, self.request.end_date,
            20_000, ("Sightseeing",)
        )
        options = get_mock_booking_options("Kochi", request.nights, 500)
        standard, standard_expenses = _calculate_profile(options, request, "standard")
        budget, budget_expenses = _calculate_profile(options, request, "budget")
        self.assertEqual(standard["profile"], "standard")
        self.assertFalse(standard_expenses["within_budget"])
        self.assertEqual(budget["profile"], "budget")
        self.assertTrue(budget_expenses["within_budget"])

    def test_crewai_tools_return_cached_evidence_and_verified_costs(self) -> None:
        tools = _make_cached_tools(
            {"provider": "Wikipedia", "results": []},
            {"available": False},
            {"distance_km": 100},
            {"notice": "Mock only"},
            self.request,
        )
        budget_result = tools[4].run(
            total_budget_inr=20_000,
            hotel_per_night_inr=2_000,
            nights=3,
            intercity_transport_inr=1_000,
            daily_local_transport_inr=300,
            daily_food_inr=600,
            daily_activities_inr=250,
        )
        self.assertIn('"total_estimate_inr": 12760.0', budget_result)
        schedule = tools[5].run(
            start_day=self.request.start_date.isoformat(),
            number_of_days=2,
            interest_list="Culture, Food",
        )
        self.assertIn("culture experience", schedule)
        self.assertIn("food option", schedule)

    def test_over_budget_plan_runs_a_second_crew_revision(self) -> None:
        request = TravelRequest(
            "Kochi", "Bengaluru", self.request.start_date, self.request.end_date,
            20_000, ("Sightseeing",)
        )
        crew = SimpleNamespace(kickoff=lambda: "coordinated plan")
        with (
            patch("travel_planner.search_destination", return_value={"results": []}),
            patch("travel_planner.get_weather_forecast", return_value={"available": False}),
            patch(
                "travel_planner.get_route_estimate",
                return_value={"distance_km": 500, "source": "https://project-osrm.org/"},
            ),
            patch("travel_planner._build_crew", return_value=crew) as build_crew,
        ):
            result = generate_travel_plan(request, "ollama", "ollama/test", self.today)

        self.assertEqual(build_crew.call_count, 2)
        self.assertTrue(result["revision_triggered"])
        self.assertFalse(result["initial_budget"]["within_budget"])
        self.assertTrue(result["budget"]["within_budget"])
        self.assertEqual(result["plan"], "coordinated plan")

    def test_forecast_outside_window_is_reported_without_api_call(self) -> None:
        start = self.today + timedelta(days=20)
        with patch("travel_planner._geocode") as geocode:
            forecast = get_weather_forecast("Kochi", start, start + timedelta(days=1), self.today)
        geocode.assert_not_called()
        self.assertFalse(forecast["available"])
        self.assertIn("within", forecast["message"])

    def test_wikipedia_search_api_results_keep_source_links(self) -> None:
        payload = {
            "query": {
                "search": [
                    {"title": "Kochi", "snippet": "Historic port city."},
                ]
            }
        }
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": ""}),
            patch("travel_planner._get_json", return_value=payload),
        ):
            result = search_destination("Kochi", ("Culture",))
        self.assertEqual(result["provider"], "Wikipedia public search API")
        self.assertEqual(
            result["results"][0]["url"], "https://en.wikipedia.org/wiki/Kochi"
        )

    def test_weather_and_routing_apis_are_normalized(self) -> None:
        forecast_start = self.today + timedelta(days=5)
        with (
            patch(
                "travel_planner._geocode",
                return_value={"name": "Kochi", "latitude": 9.9, "longitude": 76.2},
            ),
            patch(
                "travel_planner._get_json",
                return_value={
                    "daily": {"time": [forecast_start.isoformat()], "temperature_2m_max": [30]}
                },
            ),
        ):
            forecast = get_weather_forecast(
                "Kochi", forecast_start, forecast_start, self.today
            )
        self.assertTrue(forecast["available"])
        self.assertEqual(forecast["daily"]["temperature_2m_max"], [30])

        locations = [
            {"name": "Bengaluru", "latitude": 12.9, "longitude": 77.6},
            {"name": "Kochi", "latitude": 9.9, "longitude": 76.2},
        ]
        with (
            patch("travel_planner._geocode", side_effect=locations),
            patch(
                "travel_planner._get_json",
                return_value={"routes": [{"distance": 500_000, "duration": 36_000}]},
            ),
        ):
            route = get_route_estimate("Bengaluru", "Kochi")
        self.assertEqual(route["distance_km"], 500)
        self.assertEqual(route["driving_hours"], 10)


if __name__ == "__main__":
    unittest.main()
