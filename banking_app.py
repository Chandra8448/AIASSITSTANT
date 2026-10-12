from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st

from banking_agent import (
    BANKING_SERVICE,
    DEMO_ANNUAL_INTEREST_RATE,
    calculate_loan_emi,
    check_loan_eligibility,
    create_banking_agent,
    get_account_details,
    parse_loan_request,
    submit_loan_application,
)

CUSTOMERS = {
    "CUST1001 — Aarav Mehta (salary ₹60,000; existing EMI ₹5,000)": "CUST1001",
    "CUST1002 — Diya Kapoor (salary ₹35,000; existing EMI ₹8,000)": "CUST1002",
}


@st.cache_resource
def _get_agent():
    return create_banking_agent()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("text")
        )
    return str(content)


def _assess_request(customer_id: str, request: str) -> dict[str, Any]:
    amount, tenure = parse_loan_request(request)
    if amount is None or tenure is None:
        missing = []
        if amount is None:
            missing.append("loan amount")
        if tenure is None:
            missing.append("repayment term (for example, 36 months or three years)")
        raise ValueError("Please provide the " + " and ".join(missing) + ".")

    account = json.loads(get_account_details.invoke({"customer_id": customer_id}))
    emi = json.loads(
        calculate_loan_emi.invoke(
            {
                "loan_amount": amount,
                "tenure_months": tenure,
                "annual_interest_rate": DEMO_ANNUAL_INTEREST_RATE,
            }
        )
    )
    eligibility = json.loads(
        check_loan_eligibility.invoke(
            {
                "customer_id": customer_id,
                "loan_amount": amount,
                "tenure_months": tenure,
                "annual_interest_rate": DEMO_ANNUAL_INTEREST_RATE,
            }
        )
    )

    agent_text = ""
    tool_trace: list[tuple[str, str]] = []
    if os.getenv("OPENAI_API_KEY", "").strip():
        agent_result = _get_agent().invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Customer request: {request}\n"
                        f"Selected sample customer ID: {customer_id}\n"
                        f"Loan amount: ₹{amount:,}\n"
                        f"Repayment term: {tenure} months\n"
                        f"Illustrative annual interest rate: {DEMO_ANNUAL_INTEREST_RATE}%\n"
                        "Assess the request and do not submit an application.",
                    )
                ]
            }
        )
        for message in agent_result.get("messages", []):
            if getattr(message, "type", "") == "tool":
                tool_trace.append((message.name, _message_text(message.content)))
            elif getattr(message, "type", "") == "ai" and message.content:
                agent_text = _message_text(message.content)

    return {
        "customer_id": customer_id,
        "request": request,
        "loan_amount": amount,
        "tenure_months": tenure,
        "account": account,
        "emi": emi,
        "eligibility": eligibility,
        "agent_text": agent_text,
        "tool_trace": tool_trace,
        "application": None,
    }


def _render_assessment(result: dict[str, Any]) -> None:
    account = result["account"]
    eligibility = result["eligibility"]
    st.subheader("Loan assessment")
    metrics = st.columns(3)
    metrics[0].metric("Estimated monthly EMI", f"₹{result['emi']['monthly_emi_inr']:,.0f}")
    metrics[1].metric(
        "Preliminary eligibility",
        "Meets demo check" if eligibility["eligible"] else "Does not meet demo check",
    )
    metrics[2].metric(
        "Total monthly debt",
        f"₹{eligibility['total_monthly_debt_inr']:,.0f}",
        f"{eligibility['debt_to_income_ratio_pct']}% of sample salary",
    )
    st.caption(
        f"Illustrative rate: {DEMO_ANNUAL_INTEREST_RATE:.1f}% annually · "
        f"Term: {result['tenure_months']} months · "
        f"Existing EMI: ₹{account['existing_emi_inr']:,}"
    )
    with st.expander("Sample account details"):
        st.write(
            f"**{account['full_name']}** · {account['customer_id']} · "
            f"{account['account_type']} account {account['masked_account_number']}"
        )
        st.write(
            f"Illustrative balance: ₹{account['balance_inr']:,} · "
            f"Sample monthly salary: ₹{account['monthly_salary_inr']:,}"
        )
        st.caption(account["data_notice"])

    st.info(eligibility["decision"])
    if result["agent_text"]:
        st.markdown("#### Banking agent")
        st.write(result["agent_text"])
    else:
        st.caption(
            "Local demo response. Set OPENAI_API_KEY in the deployment environment "
            "to enable model-directed tool selection."
        )
    if result["tool_trace"]:
        with st.expander("Agent tool results"):
            for tool_name, tool_output in result["tool_trace"]:
                st.markdown(f"**{tool_name}**")
                st.code(tool_output, language="json")

    application = result.get("application")
    if application:
        st.success(
            f"Application {application['application_id']} submitted for review. "
            "This is not a loan approval."
        )
    elif eligibility["eligible"]:
        st.warning(
            f"Review the ₹{result['loan_amount']:,} request and estimated EMI. "
            "No application has been submitted."
        )
        if st.button("Confirm and submit loan application", type="primary"):
            try:
                code = BANKING_SERVICE.create_confirmation_code(
                    result["customer_id"],
                    result["loan_amount"],
                    result["tenure_months"],
                    DEMO_ANNUAL_INTEREST_RATE,
                )
                result["application"] = json.loads(
                    submit_loan_application.invoke(
                        {
                            "customer_id": result["customer_id"],
                            "loan_amount": result["loan_amount"],
                            "tenure_months": result["tenure_months"],
                            "annual_interest_rate": DEMO_ANNUAL_INTEREST_RATE,
                            "confirmation_code": code,
                        }
                    )
                )
                st.rerun()
            except Exception as error:
                st.error(f"The loan application could not be submitted: {error}")
    else:
        st.warning("The sample affordability check did not pass; no application can be submitted.")


def main() -> None:
    st.set_page_config(page_title="Banking Support Agent", page_icon="🏦", layout="centered")
    st.markdown(
        """
        <style>
        .stApp { background: #f5f7f2; }
        .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 4rem; }
        h1, h2, h3 { color: #202b27; }
        [data-testid="stMetric"] { background: white; border: 1px solid #dfe6df; border-radius: 8px; padding: .8rem 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### BANKING SUPPORT · TRAINING DEMO")
    st.title("Personal loan assistant")
    st.write(
        "Get an EMI estimate, review sample-account affordability, and choose "
        "whether to submit a demo application."
    )
    st.warning(
        "Fictional sample data only. The illustrative eligibility check is not "
        "a bank decision or loan approval."
    )

    with st.form("banking-request"):
        selected_customer = st.selectbox("Sample customer account", list(CUSTOMERS))
        request = st.text_area(
            "How can we help?",
            value=(
                "I want to apply for a personal loan of ₹5,00,000 for three years. "
                "My monthly salary is ₹60,000, and I already pay ₹5,000 as an EMI "
                "for another loan. Calculate my monthly EMI, check my loan eligibility, "
                "retrieve my account details, and help me submit a loan application if I qualify."
            ),
            height=150,
        )
        assess = st.form_submit_button("Assess loan request", type="primary")

    if assess:
        st.session_state.banking_result = None
        try:
            with st.spinner("Checking the sample account and loan request..."):
                st.session_state.banking_result = _assess_request(
                    CUSTOMERS[selected_customer], request
                )
            st.session_state.banking_error = None
        except ValueError as error:
            st.session_state.banking_error = str(error)
        except Exception as error:
            st.session_state.banking_error = (
                f"The banking assessment could not be completed: {error}"
            )

    error = st.session_state.get("banking_error")
    if error:
        st.info(error)
    result = st.session_state.get("banking_result")
    if result:
        _render_assessment(result)


if __name__ == "__main__":
    main()
