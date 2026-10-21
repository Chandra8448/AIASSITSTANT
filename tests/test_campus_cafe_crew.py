from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from campus_cafe_crew import (
    CafeAssumptions,
    CafeResearchError,
    _adapt_tool,
    _create_team,
    _extract_final_brief,
    _safe_calculate,
    calculate_break_even,
    calculator,
    web_search,
    wikipedia_search,
)


class CalculatorTests(unittest.TestCase):
    def test_evaluates_safe_arithmetic(self) -> None:
        self.assertEqual(_safe_calculate("(120*4.5 - 300) * 30"), 7200)
        self.assertEqual(calculator.invoke({"expression": "150000/(110*(1-0.35))"}), "2097.902098")

    def test_rejects_python_expressions_and_names(self) -> None:
        for expression in ("__import__('os').system('whoami')", "1 + x", "2 ** 100"):
            with self.subTest(expression=expression):
                with self.assertRaises(CafeResearchError):
                    _safe_calculate(expression)

    def test_calculates_break_even_with_traceable_assumptions(self) -> None:
        with patch("campus_cafe_crew.calculator", wraps=calculator) as calculator_spy:
            result = calculate_break_even()

        self.assertEqual(result["fixed_monthly_costs_inr"], 150_000)
        self.assertEqual(result["contribution_per_transaction_inr"], 71.5)
        self.assertEqual(result["monthly_break_even_transactions_rounded_up"], 2_098)
        self.assertEqual(result["daily_break_even_transactions_rounded_up"], 81)
        self.assertEqual(result["monthly_surplus_at_daily_target_inr"], 579)
        self.assertIn("3*18000.0", result["fixed_cost_expression"])
        self.assertEqual(calculator_spy.invoke.call_count, 8)

    def test_custom_assumptions_change_break_even(self) -> None:
        assumptions = CafeAssumptions(open_days_per_month=20)
        result = calculate_break_even(assumptions)

        self.assertEqual(result["daily_break_even_transactions_rounded_up"], 105)


class LangChainToolTests(unittest.TestCase):
    def test_adapted_tool_is_callable_by_crewai(self) -> None:
        adapter = _adapt_tool(calculator)

        self.assertEqual(adapter.run(query="2 + 2"), "4")

    def test_adapter_maps_query_to_tool_schema_argument(self) -> None:
        tool = Mock(name="custom_calculator", description="Test calculator")
        tool.name = "custom_calculator"
        tool.description = "Test calculator"
        tool.args = {"expression": {"type": "string"}}
        tool.invoke.return_value = "4"
        adapter = _adapt_tool(tool)

        self.assertEqual(adapter.run(query="2 + 2"), "4")
        tool.invoke.assert_called_once_with({"expression": "2 + 2"})

    @patch("campus_cafe_crew.requests.get")
    def test_scholarly_search_returns_cited_results(self, get: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "title": "Campus dining choices",
                    "publication_year": 2023,
                    "doi": "https://doi.org/10.example/campus",
                    "abstract_inverted_index": {"students": [0], "choose": [1]},
                    "primary_location": {"source": {"display_name": "Example Journal"}},
                }
            ]
        }
        get.return_value = response

        result = web_search.invoke({"query": "student food choices"})

        self.assertIn("https://doi.org/10.example/campus", result)
        self.assertIn("students choose", result)

    @patch("campus_cafe_crew.requests.get", side_effect=requests.ConnectionError("offline"))
    def test_scholarly_search_failure_is_explicit(self, _get: Mock) -> None:
        result = web_search.invoke({"query": "student café"})

        self.assertIn("Research search failed", result)
        self.assertIn("offline", result)

    @patch("campus_cafe_crew.requests.get")
    def test_wikipedia_tool_returns_title_and_link(self, get: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "query": {"search": [{"title": "Campus café", "snippet": "A place on campus."}]}
        }
        get.return_value = response

        result = wikipedia_search.invoke({"query": "campus café"})

        self.assertIn("https://en.wikipedia.org/wiki/Campus_café", result)


class CrewTests(unittest.TestCase):
    def test_crew_has_five_agents_and_sequential_contextual_tasks(self) -> None:
        with (
            patch("campus_cafe_crew._create_llm", return_value=object()),
            patch("campus_cafe_crew.Agent") as agent_factory,
            patch("campus_cafe_crew.Task") as task_factory,
            patch("campus_cafe_crew.Crew") as crew_factory,
        ):
            crew, review_task = _create_team(CafeAssumptions(), "ollama", None)

        self.assertEqual(agent_factory.call_count, 5)
        self.assertEqual(task_factory.call_count, 5)
        self.assertIs(review_task, task_factory.return_value)
        crew_args = crew_factory.call_args.kwargs
        self.assertEqual(len(crew_args["agents"]), 5)
        self.assertEqual(len(crew_args["tasks"]), 5)
        self.assertEqual(str(crew_args["process"]), "Process.sequential")
        self.assertIsNotNone(crew)

    def test_requires_explicit_final_brief_section(self) -> None:
        with self.assertRaisesRegex(CafeResearchError, "FINAL BRIEF"):
            _extract_final_brief("## Critical review\nToo much risk.")

    def test_extracts_final_brief_without_trailing_audit_section(self) -> None:
        brief = _extract_final_brief(
            "## Critical review\nThree issues fixed.\n\n"
            "## FINAL BRIEF\n# Campus Café\n\n"
            "This corrected launch brief contains enough evidence, assumptions, and "
            "arithmetic to meet the minimum length of the extraction test.\n\n"
            "## Sources and assumptions\nAudit list."
        )

        self.assertTrue(brief.startswith("# Campus Café"))
        self.assertNotIn("Audit list", brief)


if __name__ == "__main__":
    unittest.main()
