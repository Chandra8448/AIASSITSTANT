from __future__ import annotations

import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from order_management import (
    OrderManagementError,
    _agent_escalated,
    _build_crew,
    _customer_response,
    _create_llm,
    find_inventory,
    find_order,
    get_inventory_evidence,
    handle_order_query,
    requires_human_escalation,
)


class OrderDataTests(unittest.TestCase):
    def test_finds_delayed_order_and_tracking_estimate(self) -> None:
        order = find_order(" ord1025 ")

        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order.shipment_status, "delayed")
        self.assertEqual(order.revised_delivery, date(2026, 10, 17))
        self.assertEqual(order.tracking_id, "TRK-1025")

    def test_delivered_order_is_found(self) -> None:
        order = find_order("ORD1001")

        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order.shipment_status, "delivered")

    def test_out_of_stock_product_has_available_replacement(self) -> None:
        inventory = get_inventory_evidence("SH300")

        self.assertEqual(inventory["stock"], 0)
        self.assertEqual(
            inventory["available_alternatives"],
            [{"product_id": "SH301", "product_name": "Studio Headphones Lite", "stock": 7}],
        )

    def test_unknown_order_returns_no_record(self) -> None:
        self.assertIsNone(find_order("ORD9999"))

    def test_inventory_records_are_callable_as_crew_tools(self) -> None:
        inventory = find_inventory("DL100")
        assert inventory is not None
        self.assertEqual(inventory.stock, 24)


class EscalationTests(unittest.TestCase):
    def test_delayed_order_with_revised_eta_does_not_require_escalation(self) -> None:
        order = find_order("ORD1025")
        assert order is not None
        required, reasons = requires_human_escalation(
            order,
            get_inventory_evidence(order.product_id),
            {
                "found": True,
                "status": "delayed",
                "revised_delivery": "2026-10-17",
            },
            "Please check the new delivery estimate.",
        )

        self.assertFalse(required)
        self.assertEqual(reasons, [])

    def test_delivered_but_missing_package_requires_escalation(self) -> None:
        order = find_order("ORD1001")
        assert order is not None
        required, reasons = requires_human_escalation(
            order,
            get_inventory_evidence(order.product_id),
            {"found": True, "status": "delivered"},
            "It says delivered but I have not received it.",
        )

        self.assertTrue(required)
        self.assertTrue(any("marked delivered" in reason for reason in reasons))

    def test_lost_order_with_repeated_issues_requires_escalation(self) -> None:
        order = find_order("ORD1040")
        assert order is not None
        required, reasons = requires_human_escalation(
            order,
            get_inventory_evidence(order.product_id),
            {"found": True, "status": "lost"},
            "Can you check this order?",
        )

        self.assertTrue(required)
        self.assertTrue(any("lost or reported an exception" in reason for reason in reasons))
        self.assertTrue(any("prior support issues" in reason for reason in reasons))

    def test_unknown_order_requires_human_verification(self) -> None:
        required, reasons = requires_human_escalation(
            None,
            {"found": False},
            {"found": False},
            "Where is my order?",
        )

        self.assertTrue(required)
        self.assertTrue(reasons)

    def test_agent_escalation_marker_is_parsed(self) -> None:
        self.assertTrue(_agent_escalated("ESCALATION: YES\nPlease contact support."))
        self.assertFalse(_agent_escalated("ESCALATION: NO\nThe revised date is Oct 17."))
        self.assertIsNone(_agent_escalated("The package is delayed."))

    def test_customer_response_hides_escalation_marker(self) -> None:
        self.assertEqual(
            _customer_response("ESCALATION: YES\nA representative will follow up."),
            "A representative will follow up.",
        )


class WorkflowTests(unittest.TestCase):
    def test_requires_order_id_and_question(self) -> None:
        with self.assertRaisesRegex(OrderManagementError, "order ID"):
            handle_order_query("", "Where is my order?")
        with self.assertRaisesRegex(OrderManagementError, "Enter a question"):
            handle_order_query("ORD1025", " ")

    def test_ollama_is_default_and_uses_local_server(self) -> None:
        with (
            patch("order_management.LLM") as factory,
            patch.dict("os.environ", {"ORDER_OLLAMA_MODEL": "ollama/llama3.2"}, clear=True),
        ):
            _create_llm("ollama")

        factory.assert_called_once_with(
            model="ollama/llama3.2",
            base_url="http://localhost:11434",
            temperature=0.2,
        )

    def test_crew_has_four_specialists_and_sequential_tasks(self) -> None:
        order = find_order("ORD1025")
        assert order is not None
        with (
            patch("order_management._create_llm", return_value=object()),
            patch("order_management.Agent") as agent_factory,
            patch("order_management.Task") as task_factory,
            patch("order_management.Crew") as crew_factory,
        ):
            _build_crew(
                order.order_id,
                "Where is my order?",
                "ollama",
                "ollama/llama3.2",
                order,
                get_inventory_evidence(order.product_id),
                {"found": True, "status": "delayed"},
                False,
                [],
            )

        self.assertEqual(agent_factory.call_count, 4)
        self.assertEqual(task_factory.call_count, 4)
        crew_args = crew_factory.call_args.kwargs
        self.assertEqual(len(crew_args["agents"]), 4)
        self.assertEqual(len(crew_args["tasks"]), 4)
        self.assertEqual(str(crew_args["process"]), "Process.sequential")

    def test_order_query_applies_mandatory_escalation_guard(self) -> None:
        crew = SimpleNamespace(kickoff=lambda: "ESCALATION: NO\nPlease contact support.")
        with (
            patch("order_management._build_crew", return_value=crew),
            patch("order_management._kickoff", return_value=str(crew.kickoff())),
        ):
            result = handle_order_query(
                "ORD1001",
                "The tracking says delivered but I have not received it.",
            )

        self.assertTrue(result["escalation_required"])
        self.assertTrue(result["escalation_reasons"])

    def test_missing_agent_escalation_marker_fails_safe(self) -> None:
        crew = SimpleNamespace(kickoff=lambda: "The package is delayed.")
        with (
            patch("order_management._build_crew", return_value=crew),
            patch("order_management._kickoff", return_value=str(crew.kickoff())),
        ):
            result = handle_order_query("ORD1025", "What is the revised ETA?")

        self.assertTrue(result["escalation_required"])
        self.assertTrue(
            any("required escalation decision format" in reason for reason in result["escalation_reasons"])
        )


if __name__ == "__main__":
    unittest.main()
