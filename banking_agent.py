from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from langchain_core.tools import tool

ROOT = Path(__file__).parent
DEFAULT_DATABASE_PATH = ROOT / "data" / "banking_demo.sqlite3"
DEMO_ANNUAL_INTEREST_RATE = 12.0
MAX_DEBT_TO_INCOME_RATIO = 0.5
CONFIRMATION_LIFETIME = timedelta(minutes=5)

SYSTEM_PROMPT = f"""You are a careful banking support assistant for a training demo.
Use the available tools to calculate the EMI, retrieve the selected sample account,
and check preliminary affordability before discussing eligibility. The demo's
illustrative personal-loan rate is {DEMO_ANNUAL_INTEREST_RATE:.1f}% per year.
Use the salary and existing EMI returned for the sample account as the verified
demo values; do not replace them with customer-typed claims. Ask the customer for
missing loan amount, tenure, or customer ID instead of guessing.
Eligibility is only a preliminary demo estimate, not loan approval.
Never submit an application based on a request to help, a conditional request,
or an inferred consent. The submit tool requires a short-lived authorization
code that is issued only by the app after the customer presses its explicit
confirmation button. Do not invent, guess, or ask the customer for this code.
Keep the final response concise and summarize the EMI, preliminary eligibility,
and application status without claiming approval."""


def _validate_loan_inputs(
    loan_amount: int,
    tenure_months: int,
    annual_interest_rate: float,
) -> None:
    if not isinstance(loan_amount, int) or isinstance(loan_amount, bool) or loan_amount <= 0:
        raise ValueError("Loan amount must be a positive whole number of rupees.")
    if (
        not isinstance(tenure_months, int)
        or isinstance(tenure_months, bool)
        or not 1 <= tenure_months <= 360
    ):
        raise ValueError("Tenure must be between 1 and 360 months.")
    if (
        not isinstance(annual_interest_rate, (int, float))
        or isinstance(annual_interest_rate, bool)
        or not math.isfinite(annual_interest_rate)
        or not 0 <= annual_interest_rate <= 100
    ):
        raise ValueError("Annual interest rate must be between 0 and 100 percent.")


def _calculate_emi(
    loan_amount: int,
    tenure_months: int,
    annual_interest_rate: float,
) -> float:
    _validate_loan_inputs(loan_amount, tenure_months, annual_interest_rate)
    monthly_rate = annual_interest_rate / 12 / 100
    if monthly_rate == 0:
        return round(loan_amount / tenure_months, 2)
    factor = (1 + monthly_rate) ** tenure_months
    return round(loan_amount * monthly_rate * factor / (factor - 1), 2)


def parse_loan_request(request: str) -> tuple[int | None, int | None]:
    """Extract an explicitly stated rupee amount and term from a request."""
    lakh_match = re.search(
        r"(?:loan(?:\s+(?:of|amount(?:\s+of)?))?\s*)?₹?\s*([\d.]+)\s*(?:lakh|lakhs|lac|lacs)\b",
        request,
        re.IGNORECASE,
    )
    if lakh_match:
        amount_inr = round(float(lakh_match.group(1)) * 100_000)
    else:
        amount_match = re.search(
            r"(?:personal\s+)?loan(?:\s+(?:of|amount(?:\s+of)?))?\s*(?:₹|rs\.?\s*)?([\d,]+)",
            request,
            re.IGNORECASE,
        )
        amount_inr = (
            int(amount_match.group(1).replace(",", "")) if amount_match else None
        )

    number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    term_match = re.search(
        r"\b(\d+|one|two|three|four|five)\s*(years?|yrs?|months?|mos?)\b",
        request,
        re.IGNORECASE,
    )
    tenure_months = None
    if term_match:
        quantity = number_words.get(term_match.group(1).lower())
        if quantity is None:
            quantity = int(term_match.group(1))
        unit = term_match.group(2).lower()
        tenure_months = quantity * 12 if unit.startswith(("year", "yr")) else quantity
    return amount_inr, tenure_months


class BankingService:
    """SQLite-backed sample banking data and application workflow."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    masked_account_number TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    balance_inr INTEGER NOT NULL,
                    monthly_salary_inr INTEGER NOT NULL,
                    existing_emi_inr INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS loan_applications (
                    application_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                    loan_amount_inr INTEGER NOT NULL,
                    tenure_months INTEGER NOT NULL,
                    annual_interest_rate REAL NOT NULL,
                    monthly_emi_inr REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_confirmations (
                    confirmation_hash TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                    loan_amount_inr INTEGER NOT NULL,
                    tenure_months INTEGER NOT NULL,
                    annual_interest_rate REAL NOT NULL,
                    expires_at TEXT NOT NULL
                );
                """
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO customers
                    (customer_id, full_name, masked_account_number, account_type,
                     balance_inr, monthly_salary_inr, existing_emi_inr)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("CUST1001", "Aarav Mehta", "XXXXXX4821", "Savings", 84500, 60000, 5000),
                    ("CUST1002", "Diya Kapoor", "XXXXXX7394", "Savings", 39200, 35000, 8000),
                ],
            )

    def _get_customer(self, customer_id: str) -> dict[str, Any]:
        self._initialize()
        with self._connection() as connection:
            customer = connection.execute(
                "SELECT * FROM customers WHERE customer_id = ?",
                (customer_id.strip().upper(),),
            ).fetchone()
        if customer is None:
            raise ValueError("Unknown sample customer ID. Choose CUST1001 or CUST1002.")
        return dict(customer)

    def get_account_details(self, customer_id: str) -> dict[str, Any]:
        customer = self._get_customer(customer_id)
        return {
            "customer_id": customer["customer_id"],
            "full_name": customer["full_name"],
            "masked_account_number": customer["masked_account_number"],
            "account_type": customer["account_type"],
            "balance_inr": customer["balance_inr"],
            "monthly_salary_inr": customer["monthly_salary_inr"],
            "existing_emi_inr": customer["existing_emi_inr"],
            "data_notice": "Illustrative training data only; not a real bank account.",
        }

    def check_loan_eligibility(
        self,
        customer_id: str,
        loan_amount: int,
        tenure_months: int,
        annual_interest_rate: float = DEMO_ANNUAL_INTEREST_RATE,
    ) -> dict[str, Any]:
        _validate_loan_inputs(loan_amount, tenure_months, annual_interest_rate)
        customer = self._get_customer(customer_id)
        emi = _calculate_emi(loan_amount, tenure_months, annual_interest_rate)
        total_monthly_debt = customer["existing_emi_inr"] + emi
        debt_to_income_ratio = total_monthly_debt / customer["monthly_salary_inr"]
        eligible = debt_to_income_ratio <= MAX_DEBT_TO_INCOME_RATIO
        return {
            "customer_id": customer["customer_id"],
            "loan_amount_inr": loan_amount,
            "tenure_months": tenure_months,
            "annual_interest_rate_pct": annual_interest_rate,
            "monthly_emi_inr": emi,
            "monthly_salary_inr": customer["monthly_salary_inr"],
            "existing_emi_inr": customer["existing_emi_inr"],
            "total_monthly_debt_inr": round(total_monthly_debt, 2),
            "debt_to_income_ratio_pct": round(debt_to_income_ratio * 100, 2),
            "maximum_debt_to_income_ratio_pct": MAX_DEBT_TO_INCOME_RATIO * 100,
            "eligible": eligible,
            "decision": (
                "Meets the demo affordability check; this is not loan approval."
                if eligible
                else "Does not meet the demo affordability check; this is not a bank decision."
            ),
        }

    def create_confirmation_code(
        self,
        customer_id: str,
        loan_amount: int,
        tenure_months: int,
        annual_interest_rate: float = DEMO_ANNUAL_INTEREST_RATE,
    ) -> str:
        eligibility = self.check_loan_eligibility(
            customer_id, loan_amount, tenure_months, annual_interest_rate
        )
        if not eligibility["eligible"]:
            raise ValueError("An application can only be submitted after the demo eligibility check passes.")

        code = secrets.token_urlsafe(32)
        expires_at = (datetime.now(UTC) + CONFIRMATION_LIFETIME).isoformat()
        self._initialize()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO pending_confirmations
                    (confirmation_hash, customer_id, loan_amount_inr,
                     tenure_months, annual_interest_rate, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    eligibility["customer_id"],
                    loan_amount,
                    tenure_months,
                    annual_interest_rate,
                    expires_at,
                ),
            )
        return code

    def submit_loan_application(
        self,
        customer_id: str,
        loan_amount: int,
        tenure_months: int,
        annual_interest_rate: float,
        confirmation_code: str,
    ) -> dict[str, Any]:
        _validate_loan_inputs(loan_amount, tenure_months, annual_interest_rate)
        if not confirmation_code:
            raise ValueError("Explicit customer confirmation is required before submitting an application.")

        normalized_customer_id = customer_id.strip().upper()
        confirmation_hash = hashlib.sha256(confirmation_code.encode("utf-8")).hexdigest()
        self._initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pending = connection.execute(
                """
                SELECT * FROM pending_confirmations
                WHERE confirmation_hash = ? AND customer_id = ?
                  AND loan_amount_inr = ? AND tenure_months = ?
                  AND annual_interest_rate = ?
                """,
                (
                    confirmation_hash,
                    normalized_customer_id,
                    loan_amount,
                    tenure_months,
                    annual_interest_rate,
                ),
            ).fetchone()
            if pending is None:
                raise ValueError("Confirmation is missing, expired, or does not match this loan request.")
            if datetime.fromisoformat(pending["expires_at"]) <= datetime.now(UTC):
                connection.execute(
                    "DELETE FROM pending_confirmations WHERE confirmation_hash = ?",
                    (confirmation_hash,),
                )
                raise ValueError("Confirmation expired. Review the application details and confirm again.")

            customer = connection.execute(
                "SELECT * FROM customers WHERE customer_id = ?",
                (normalized_customer_id,),
            ).fetchone()
            if customer is None:
                raise ValueError("Unknown sample customer ID.")
            emi = _calculate_emi(loan_amount, tenure_months, annual_interest_rate)
            total_debt = customer["existing_emi_inr"] + emi
            if total_debt / customer["monthly_salary_inr"] > MAX_DEBT_TO_INCOME_RATIO:
                raise ValueError("The application no longer meets the demo affordability check.")

            application_id = f"DEMO-{uuid.uuid4().hex[:12].upper()}"
            connection.execute(
                """
                INSERT INTO loan_applications
                    (application_id, customer_id, loan_amount_inr, tenure_months,
                     annual_interest_rate, monthly_emi_inr, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application_id,
                    normalized_customer_id,
                    loan_amount,
                    tenure_months,
                    annual_interest_rate,
                    emi,
                    "submitted_for_review",
                    datetime.now(UTC).isoformat(),
                ),
            )
            connection.execute(
                "DELETE FROM pending_confirmations WHERE confirmation_hash = ?",
                (confirmation_hash,),
            )
        return {
            "application_id": application_id,
            "customer_id": normalized_customer_id,
            "loan_amount_inr": loan_amount,
            "tenure_months": tenure_months,
            "monthly_emi_inr": emi,
            "status": "submitted_for_review",
            "notice": "Demo application submitted for review; this is not a loan approval.",
        }


BANKING_SERVICE = BankingService()


@tool
def calculate_loan_emi(
    loan_amount: int,
    tenure_months: int,
    annual_interest_rate: float = DEMO_ANNUAL_INTEREST_RATE,
) -> str:
    """Calculate the monthly EMI for a rupee loan, term in months, and annual rate in percent."""
    return json.dumps(
        {
            "loan_amount_inr": loan_amount,
            "tenure_months": tenure_months,
            "annual_interest_rate_pct": annual_interest_rate,
            "monthly_emi_inr": _calculate_emi(
                loan_amount, tenure_months, annual_interest_rate
            ),
            "rate_notice": "Illustrative sample-bank rate; confirm the actual offer with the bank.",
        }
    )


@tool
def check_loan_eligibility(
    customer_id: str,
    loan_amount: int,
    tenure_months: int,
    annual_interest_rate: float = DEMO_ANNUAL_INTEREST_RATE,
) -> str:
    """Check preliminary affordability using a sample customer's verified salary and existing EMI."""
    return json.dumps(
        BANKING_SERVICE.check_loan_eligibility(
            customer_id, loan_amount, tenure_months, annual_interest_rate
        )
    )


@tool
def get_account_details(customer_id: str) -> str:
    """Retrieve masked account details and sample salary/current EMI for a demo customer ID."""
    return json.dumps(BANKING_SERVICE.get_account_details(customer_id))


@tool
def submit_loan_application(
    customer_id: str,
    loan_amount: int,
    tenure_months: int,
    annual_interest_rate: float,
    confirmation_code: str,
) -> str:
    """Submit an eligible application only with the one-time code issued after explicit UI confirmation."""
    return json.dumps(
        BANKING_SERVICE.submit_loan_application(
            customer_id,
            loan_amount,
            tenure_months,
            annual_interest_rate,
            confirmation_code,
        )
    )


BANKING_TOOLS = [
    calculate_loan_emi,
    check_loan_eligibility,
    get_account_details,
    submit_loan_application,
]


def create_banking_agent():
    """Create the single LangGraph tool-calling agent; an OpenAI key is required."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY to use the AI banking agent.")
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=api_key,
    )
    return create_react_agent(model, BANKING_TOOLS, prompt=SYSTEM_PROMPT)
