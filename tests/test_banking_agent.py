from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from banking_agent import BankingService, calculate_loan_emi, parse_loan_request


class BankingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.service = BankingService(Path(self.temp_directory.name) / "banking.sqlite3")

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_calculates_emi_for_sample_request(self) -> None:
        result = json.loads(
            calculate_loan_emi.invoke(
                {
                    "loan_amount": 500_000,
                    "tenure_months": 36,
                    "annual_interest_rate": 12.0,
                }
            )
        )

        self.assertEqual(result["monthly_emi_inr"], 16607.15)
        self.assertIn("Illustrative", result["rate_notice"])

    def test_sample_customer_meets_affordability_check(self) -> None:
        result = self.service.check_loan_eligibility("CUST1001", 500_000, 36)

        self.assertTrue(result["eligible"])
        self.assertEqual(result["monthly_salary_inr"], 60_000)
        self.assertEqual(result["existing_emi_inr"], 5_000)
        self.assertLessEqual(result["debt_to_income_ratio_pct"], 50)

    def test_unaffordable_application_cannot_be_confirmed_or_submitted(self) -> None:
        result = self.service.check_loan_eligibility("CUST1002", 500_000, 36)

        self.assertFalse(result["eligible"])
        with self.assertRaisesRegex(ValueError, "only be submitted"):
            self.service.create_confirmation_code("CUST1002", 500_000, 36)

    def test_application_requires_explicit_one_time_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Confirmation is missing"):
            self.service.submit_loan_application(
                "CUST1001", 500_000, 36, 12.0, "not-a-confirmation-code"
            )

        confirmation_code = self.service.create_confirmation_code("CUST1001", 500_000, 36)
        application = self.service.submit_loan_application(
            "CUST1001", 500_000, 36, 12.0, confirmation_code
        )
        self.assertEqual(application["status"], "submitted_for_review")

        with self.assertRaisesRegex(ValueError, "Confirmation is missing"):
            self.service.submit_loan_application(
                "CUST1001", 500_000, 36, 12.0, confirmation_code
            )

    def test_confirmation_is_bound_to_the_reviewed_request(self) -> None:
        confirmation_code = self.service.create_confirmation_code("CUST1001", 500_000, 36)

        with self.assertRaisesRegex(ValueError, "does not match"):
            self.service.submit_loan_application(
                "CUST1001", 400_000, 36, 12.0, confirmation_code
            )

    def test_loan_request_parser_extracts_amount_and_term(self) -> None:
        self.assertEqual(
            parse_loan_request("I need a loan of ₹5,00,000 for three years."),
            (500_000, 36),
        )
        self.assertEqual(
            parse_loan_request("Please lend me ₹5 lakh over 36 months."),
            (500_000, 36),
        )
        self.assertEqual(parse_loan_request("Can I borrow?"), (None, None))


if __name__ == "__main__":
    unittest.main()
