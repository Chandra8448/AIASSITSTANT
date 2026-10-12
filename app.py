from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict

import chromadb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from banking_agent import (
    BANKING_SERVICE,
    DEMO_ANNUAL_INTEREST_RATE,
    calculate_loan_emi,
    check_loan_eligibility,
    create_banking_agent,
    get_account_details,
    submit_loan_application,
)

load_dotenv()

ROOT = Path(__file__).parent
CATALOG_PATH = ROOT / "data" / "product_catalog.csv"
FAQ_PATH = ROOT / "data" / "product_faq.csv"
CHROMA_PATH = ROOT / ".chroma"
DEFAULT_BUDGET = 60000


class ShoppingState(TypedDict, total=False):
    request: str
    criteria: dict[str, Any]
    search_results: list[dict[str, Any]]
    inventory_results: list[dict[str, Any]]
    budget_results: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]


def load_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(CATALOG_PATH)
    required = {
        "product_id", "name", "brand", "price_inr", "processor", "ram_gb",
        "storage", "battery_hours", "stock", "delivery_days", "discount_pct",
        "weight_kg", "graphics", "best_for", "description",
    }
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {', '.join(sorted(missing))}")
    return catalog


def parse_requirements(request: str) -> dict[str, Any]:
    budget_match = re.search(
        r"(?:under|below|upto|up to|max(?:imum)?|budget(?: of)?|within|₹|rs\.?\s*)\s*₹?\s*([\d,]+)",
        request,
        re.IGNORECASE,
    )
    ram_match = re.search(r"(\d+)\s*gb\s*(?:ram|memory)", request, re.IGNORECASE)
    return {
        "budget_inr": int(budget_match.group(1).replace(",", "")) if budget_match else DEFAULT_BUDGET,
        "min_ram_gb": int(ram_match.group(1)) if ram_match else 8,
        "battery_priority": bool(re.search(r"battery|portable|travel", request, re.IGNORECASE)),
        "request_terms": re.findall(r"[a-z0-9]+", request.lower()),
    }


def _catalog_records() -> list[dict[str, Any]]:
    return load_catalog().to_dict(orient="records")


@tool("search_product_catalog")
def search_product_catalog(request: str) -> str:
    """Search the local laptop catalog against a customer's budget and RAM needs."""
    criteria = parse_requirements(request)
    products = _catalog_records()
    exact = [
        product for product in products
        if product["price_inr"] <= criteria["budget_inr"]
        and product["ram_gb"] >= criteria["min_ram_gb"]
    ]
    candidates = exact or sorted(
        products,
        key=lambda product: (
            product["price_inr"] > criteria["budget_inr"],
            product["ram_gb"] < criteria["min_ram_gb"],
            abs(product["price_inr"] - criteria["budget_inr"]),
        ),
    )[:5]
    return json.dumps(candidates)


@tool("check_product_inventory")
def check_product_inventory(product_id: str) -> str:
    """Check local stock and estimated delivery for a product ID."""
    matches = [item for item in _catalog_records() if item["product_id"] == product_id]
    if not matches:
        return json.dumps({"product_id": product_id, "in_stock": False, "stock": 0})
    product = matches[0]
    return json.dumps({
        "product_id": product_id,
        "in_stock": product["stock"] > 0,
        "stock": product["stock"],
        "delivery_days": product["delivery_days"],
    })


@tool("validate_shopping_budget")
def validate_shopping_budget(price_inr: int, discount_pct: int, budget_inr: int) -> str:
    """Apply the catalog discount and report whether the final price fits the budget."""
    final_price = round(price_inr * (100 - discount_pct) / 100)
    return json.dumps({
        "list_price_inr": price_inr,
        "discount_pct": discount_pct,
        "final_price_inr": final_price,
        "within_budget": final_price <= budget_inr,
        "budget_inr": budget_inr,
    })


def _search_agent(state: ShoppingState) -> dict[str, Any]:
    criteria = parse_requirements(state["request"])
    candidates = json.loads(search_product_catalog.invoke({"request": state["request"]}))
    return {"criteria": criteria, "search_results": candidates}


def _inventory_agent(state: ShoppingState) -> dict[str, Any]:
    inventory_checked = []
    for product in state.get("search_results", []):
        inventory = json.loads(check_product_inventory.invoke({"product_id": product["product_id"]}))
        inventory_checked.append({**product, **inventory})
    return {"inventory_results": inventory_checked}


def _budget_agent(state: ShoppingState) -> dict[str, Any]:
    budget = state["criteria"]["budget_inr"]
    checked = []
    for product in state.get("inventory_results", []):
        verdict = json.loads(validate_shopping_budget.invoke({
            "price_inr": int(product["price_inr"]),
            "discount_pct": int(product["discount_pct"]),
            "budget_inr": budget,
        }))
        checked.append({**product, **verdict})
    return {"budget_results": checked}


def _recommendation_agent(state: ShoppingState) -> dict[str, Any]:
    criteria = state["criteria"]
    request_terms = set(criteria["request_terms"])
    available = [
        product for product in state.get("budget_results", [])
        if product["in_stock"] and product["within_budget"]
    ]
    for product in available:
        best_for = set(re.findall(r"[a-z0-9]+", str(product["best_for"]).lower()))
        score = 50
        score += 20 if product["ram_gb"] >= criteria["min_ram_gb"] else -18
        score += min(product["battery_hours"] * 2, 20) if criteria["battery_priority"] else 8
        score += min(len(request_terms & best_for) * 5, 15)
        score += 10 if product["graphics"] == "Dedicated" and "gaming" in request_terms else 0
        product["fit_score"] = min(score, 100)
        product["why_recommended"] = _explanation(product, criteria)
    available.sort(key=lambda item: (-item["fit_score"], item["final_price_inr"]))
    return {"recommendations": available}


def _explanation(product: dict[str, Any], criteria: dict[str, Any]) -> str:
    reasons = [f"{product['ram_gb']} GB RAM meets your {criteria['min_ram_gb']} GB preference"]
    if criteria["battery_priority"]:
        reasons.append(f"{product['battery_hours']}-hour rated battery supports longer sessions")
    reasons.append(f"the discounted price of ₹{product['final_price_inr']:,} stays within budget")
    return "; ".join(reasons) + "."


@st.cache_resource
def build_workflow():
    graph = StateGraph(ShoppingState)
    graph.add_node("product_search_agent", _search_agent)
    graph.add_node("inventory_agent", _inventory_agent)
    graph.add_node("budget_agent", _budget_agent)
    graph.add_node("recommendation_agent", _recommendation_agent)
    graph.add_edge(START, "product_search_agent")
    graph.add_edge("product_search_agent", "inventory_agent")
    graph.add_edge("inventory_agent", "budget_agent")
    graph.add_edge("budget_agent", "recommendation_agent")
    graph.add_edge("recommendation_agent", END)
    return graph.compile()


def _embedding(text: str, dimensions: int = 256) -> list[float]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / magnitude for value in vector]


@st.cache_resource
def get_faq_collection():
    faq = pd.read_csv(FAQ_PATH).fillna("")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name="shopping_product_faq",
        metadata={"hnsw:space": "cosine"},
    )
    if not faq.empty:
        documents = [f"Question: {row.question}\nAnswer: {row.answer}" for row in faq.itertuples()]
        collection.upsert(
            ids=[str(row.faq_id) for row in faq.itertuples()],
            documents=documents,
            metadatas=[{"topic": str(row.topic)} for row in faq.itertuples()],
            embeddings=[_embedding(document) for document in documents],
        )
    return collection


def retrieve_faq_answer(question: str) -> tuple[str, list[str]]:
    collection = get_faq_collection()
    result = collection.query(
        query_embeddings=[_embedding(question)],
        n_results=min(3, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]
    if not documents or (distances and distances[0] > 0.9):
        return "I couldn't find a close match in the product FAQ. Try asking about warranty, returns, delivery, battery, or upgrades.", []
    citations = [str(document) for document in documents[:2]]
    context = "\n".join(f"- {document}" for document in citations)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            from langchain_openai import ChatOpenAI

            response = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0).invoke([
                ("system", "Answer the customer's product FAQ using only the supplied catalog FAQ context. If it does not contain the answer, say so. Keep the answer concise."),
                ("human", f"FAQ context:\n{context}\n\nQuestion: {question}"),
            ])
            return str(response.content), citations
        except Exception:
            pass
    answer_match = re.search(r"\nAnswer:\s*(.+)", citations[0], re.DOTALL)
    return (answer_match.group(1).strip() if answer_match else citations[0]), citations


def _format_inr(value: int | float) -> str:
    return f"₹{value:,.0f}"


def _get_local_explanation(product: dict[str, Any], request: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return product["why_recommended"]
    try:
        from langchain_openai import ChatOpenAI

        response = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0).invoke([
            ("system", "Explain this laptop recommendation using only the supplied facts. Do not invent specifications or make purchase claims. Use at most two sentences."),
            ("human", f"Customer request: {request}\nProduct facts: {json.dumps(product)}"),
        ])
        return str(response.content)
    except Exception:
        return product["why_recommended"]


def _render_product_card(product: dict[str, Any], recommended_id: str) -> None:
    badge = "BEST MATCH" if product["product_id"] == recommended_id else "AVAILABLE"
    with st.container(border=True):
        top = st.columns([4, 1])
        top[0].markdown(f"#### {product['name']}\n{product['brand']} · {product['processor']}")
        top[1].markdown(f"<span class='match-badge'>{badge}</span>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='price-line'>{_format_inr(product['final_price_inr'])} "
            f"<span class='old-price'>{_format_inr(product['list_price_inr'])}</span> "
            f"<span class='discount'>{product['discount_pct']}% off</span></div>",
            unsafe_allow_html=True,
        )
        spec_cols = st.columns(4)
        for column, label, value in zip(
            spec_cols,
            ["MEMORY", "STORAGE", "BATTERY", "DELIVERY"],
            [f"{product['ram_gb']} GB", product["storage"], f"{product['battery_hours']} hrs", f"{product['delivery_days']} days"],
        ):
            column.markdown(f"<div class='spec-label'>{label}</div><div class='spec-value'>{value}</div>", unsafe_allow_html=True)
        st.caption(product["description"])
        if product["product_id"] == recommended_id:
            st.markdown(f"<div class='reason'>{product['display_explanation']}</div>", unsafe_allow_html=True)
        if st.button("Choose this laptop", key=f"choose-{product['product_id']}", use_container_width=True):
            st.session_state.pending_product = product


def _render_cart() -> None:
    st.sidebar.markdown("## Your cart")
    cart = st.session_state.get("cart", {})
    if not cart:
        st.sidebar.caption("No items yet. Choose a laptop and confirm to add it here.")
        return
    catalog = load_catalog().set_index("product_id")
    total = 0
    for product_id, quantity in list(cart.items()):
        if product_id not in catalog.index:
            continue
        product = catalog.loc[product_id]
        discounted = round(product.price_inr * (100 - product.discount_pct) / 100)
        total += discounted * quantity
        row = st.sidebar.columns([4, 1])
        row[0].markdown(f"**{product['name']}**  \n{quantity} × {_format_inr(discounted)}")
        if row[1].button("Remove", key=f"remove-{product_id}"):
            del cart[product_id]
            st.rerun()
    st.sidebar.markdown(f"**Estimated total**  {_format_inr(total)}")
    st.sidebar.caption("Simulated cart only. No order or payment is placed.")


def _render_confirmation() -> None:
    product = st.session_state.get("pending_product")
    if not product:
        return
    st.divider()
    st.warning(f"Add **{product['name']}** to your simulated cart for {_format_inr(product['final_price_inr'])}?")
    confirm, cancel, _ = st.columns([1.5, 1, 5])
    if confirm.button("Confirm add to cart", type="primary"):
        cart = st.session_state.setdefault("cart", {})
        cart[product["product_id"]] = cart.get(product["product_id"], 0) + 1
        st.session_state.pending_product = None
        st.toast("Added to your simulated cart")
        st.rerun()
    if cancel.button("Cancel"):
        st.session_state.pending_product = None
        st.rerun()


def _parse_banking_request(request: str) -> tuple[int | None, int | None]:
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


@st.cache_resource
def _get_banking_agent():
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


def _render_banking_assistant() -> None:
    st.markdown("### Smart banking support")
    st.caption(
        "A single LangGraph agent coordinates LangChain banking tools. "
        "This training demo uses fictional sample customers and a 12% illustrative rate."
    )
    customer_labels = {
        "CUST1001 — Aarav Mehta (salary ₹60,000; existing EMI ₹5,000)": "CUST1001",
        "CUST1002 — Diya Kapoor (salary ₹35,000; existing EMI ₹8,000)": "CUST1002",
    }
    with st.form("banking-request"):
        selected_customer = st.selectbox("Sample customer account", list(customer_labels))
        request = st.text_area(
            "What would you like help with?",
            value=(
                "I want to apply for a personal loan of ₹5,00,000 for three years. "
                "My monthly salary is ₹60,000, and I already pay ₹5,000 as an EMI "
                "for another loan. Calculate my monthly EMI, check my loan eligibility, "
                "retrieve my account details, and help me submit a loan application if I qualify."
            ),
            height=150,
        )
        assess = st.form_submit_button("Assess my loan request", type="primary")

    if assess:
        st.session_state.banking_result = None
        amount_inr, tenure_months = _parse_banking_request(request)
        missing = []
        if amount_inr is None:
            missing.append("loan amount")
        if tenure_months is None:
            missing.append("repayment term (for example, 36 months or three years)")
        if missing:
            st.session_state.banking_missing_info = (
                "Please provide the " + " and ".join(missing) + " before I can assess the request."
            )
        else:
            st.session_state.banking_missing_info = None
            customer_id = customer_labels[selected_customer]
            agent_text = ""
            tool_trace: list[tuple[str, str]] = []
            try:
                if os.getenv("OPENAI_API_KEY", "").strip():
                    with st.spinner("The banking agent is coordinating the requested tools..."):
                        agent_result = _get_banking_agent().invoke(
                            {
                                "messages": [
                                    (
                                        "user",
                                        f"Customer request: {request}\n"
                                        f"Selected sample customer ID: {customer_id}\n"
                                        f"Loan amount: ₹{amount_inr:,}\n"
                                        f"Repayment term: {tenure_months} months\n"
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
                else:
                    st.info(
                        "OPENAI_API_KEY is not set, so this run uses the same local banking tools "
                        "in a deterministic demo flow. Add a key to enable model-directed tool selection."
                    )

                account = json.loads(
                    get_account_details.invoke({"customer_id": customer_id})
                )
                emi = json.loads(
                    calculate_loan_emi.invoke(
                        {
                            "loan_amount": amount_inr,
                            "tenure_months": tenure_months,
                            "annual_interest_rate": DEMO_ANNUAL_INTEREST_RATE,
                        }
                    )
                )
                eligibility = json.loads(
                    check_loan_eligibility.invoke(
                        {
                            "customer_id": customer_id,
                            "loan_amount": amount_inr,
                            "tenure_months": tenure_months,
                            "annual_interest_rate": DEMO_ANNUAL_INTEREST_RATE,
                        }
                    )
                )
                st.session_state.banking_result = {
                    "customer_id": customer_id,
                    "request": request,
                    "loan_amount": amount_inr,
                    "tenure_months": tenure_months,
                    "account": account,
                    "emi": emi,
                    "eligibility": eligibility,
                    "agent_text": agent_text,
                    "tool_trace": tool_trace,
                    "application": None,
                }
            except Exception as error:
                st.error(f"The banking assessment could not be completed: {error}")

    missing_info = st.session_state.get("banking_missing_info")
    if missing_info:
        st.info(missing_info)

    result = st.session_state.get("banking_result")
    if not result:
        return

    account = result["account"]
    eligibility = result["eligibility"]
    st.markdown("#### Assessment summary")
    metrics = st.columns(3)
    metrics[0].metric("Estimated monthly EMI", _format_inr(result["emi"]["monthly_emi_inr"]))
    metrics[1].metric(
        "Preliminary eligibility",
        "Meets demo check" if eligibility["eligible"] else "Does not meet demo check",
    )
    metrics[2].metric(
        "Total monthly debt",
        _format_inr(eligibility["total_monthly_debt_inr"]),
        f"{eligibility['debt_to_income_ratio_pct']}% of sample salary",
    )
    st.caption(
        f"Rate: {DEMO_ANNUAL_INTEREST_RATE:.1f}% illustrative annual rate · "
        f"Term: {result['tenure_months']} months · "
        f"Existing EMI: {_format_inr(account['existing_emi_inr'])}"
    )
    with st.expander("Retrieved sample account details"):
        st.write(
            f"**{account['full_name']}** · {account['customer_id']} · "
            f"{account['account_type']} account {account['masked_account_number']}"
        )
        st.write(
            f"Illustrative balance: {_format_inr(account['balance_inr'])} · "
            f"Sample monthly salary: {_format_inr(account['monthly_salary_inr'])}"
        )
        st.caption(account["data_notice"])
    st.info(eligibility["decision"])

    if result["agent_text"]:
        st.markdown("#### Banking agent")
        st.write(result["agent_text"])
    elif not os.getenv("OPENAI_API_KEY", "").strip():
        st.markdown(
            "**Demo response:** The sample account and preliminary affordability "
            "check are shown above. This estimate is not a loan approval."
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
            f"Review the ₹{result['loan_amount']:,} request and estimated EMI above. "
            "No application has been submitted."
        )
        if st.button(
            "Confirm and submit loan application",
            type="primary",
            key=f"confirm-loan-{result['customer_id']}",
        ):
            try:
                confirmation_code = BANKING_SERVICE.create_confirmation_code(
                    result["customer_id"],
                    result["loan_amount"],
                    result["tenure_months"],
                    DEMO_ANNUAL_INTEREST_RATE,
                )
                application = json.loads(
                    submit_loan_application.invoke(
                        {
                            "customer_id": result["customer_id"],
                            "loan_amount": result["loan_amount"],
                            "tenure_months": result["tenure_months"],
                            "annual_interest_rate": DEMO_ANNUAL_INTEREST_RATE,
                            "confirmation_code": confirmation_code,
                        }
                    )
                )
                result["application"] = application
                st.rerun()
            except Exception as error:
                st.error(f"The loan application could not be submitted: {error}")
    else:
        st.warning("The sample affordability check did not pass; no application can be submitted.")


def _render_results(request: str) -> None:
    result = st.session_state.get("shopping_result")
    if st.session_state.get("shopping_result_request") != request or result is None:
        with st.spinner("Product Search, Inventory, Budget and Recommendation agents are working..."):
            try:
                result = build_workflow().invoke({"request": request})
            except Exception as error:
                st.error(f"The shopping workflow could not finish: {error}")
                return
        st.session_state.shopping_result_request = request
        st.session_state.shopping_result = result

    criteria = result["criteria"]
    recommendations = result.get("recommendations", [])
    st.markdown("### Your shortlist")
    st.caption(
        f"Budget {_format_inr(criteria['budget_inr'])} · At least {criteria['min_ram_gb']} GB RAM · "
        f"{len(recommendations)} in-stock option(s) within budget"
    )
    if not recommendations:
        st.info("No in-stock laptop in the sample catalog meets both your budget and RAM requirements. Try a higher budget or lower RAM requirement.")
        near_matches = result.get("budget_results", [])
        if near_matches:
            st.markdown("#### Closest alternatives")
            st.dataframe(pd.DataFrame([
                {
                    "Product": item["name"],
                    "Price after discount": _format_inr(item["final_price_inr"]),
                    "RAM": f"{item['ram_gb']} GB",
                    "Stock": "In stock" if item["in_stock"] else "Unavailable",
                    "Budget": "Within budget" if item["within_budget"] else "Over budget",
                }
                for item in near_matches
            ]), hide_index=True, use_container_width=True)
        return

    best = recommendations[0]
    if "display_explanation" not in best:
        best["display_explanation"] = _get_local_explanation(best, request)
    columns = st.columns(3)
    columns[0].metric("Best match", best["name"])
    columns[1].metric("Price after discount", _format_inr(best["final_price_inr"]))
    columns[2].metric("Estimated delivery", f"{best['delivery_days']} days")
    st.markdown("#### Side-by-side comparison")
    comparison = pd.DataFrame([
        {
            "Product": item["name"],
            "Price": _format_inr(item["final_price_inr"]),
            "RAM": f"{item['ram_gb']} GB",
            "Processor": item["processor"],
            "Storage": item["storage"],
            "Battery": f"{item['battery_hours']} hrs",
            "Stock": f"{item['stock']} available",
            "Budget": "Within budget" if item["within_budget"] else "Over budget",
        }
        for item in recommendations
    ])
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.markdown("#### Recommended for you")
    for item in recommendations:
        item["display_explanation"] = best["display_explanation"] if item is best else item["why_recommended"]
        _render_product_card(item, best["product_id"])
    _render_confirmation()


def main() -> None:
    st.set_page_config(page_title="Common Ground | Shopping Assistant", page_icon="⌂", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root { --ink: #202b27; --muted: #66746d; --green: #1b6952; --lime: #d7ef72; --line: #dfe6df; --paper: #f5f7f2; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
        .stApp { background: var(--paper); }
        [data-testid="stHeader"] { background: rgba(245,247,242,.92); }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
        h1, h2, h3, h4 { font-family: 'Manrope', sans-serif; letter-spacing: 0; color: var(--ink); }
        h1 { font-size: 2.35rem; font-weight: 800; margin-bottom: .25rem; }
        h2 { font-size: 1.55rem; }
        .eyebrow { color: var(--green); font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
        .masthead { display: flex; align-items: center; gap: .7rem; margin-bottom: 2.1rem; }
        .brand-mark { background: var(--green); color: white; width: 35px; height: 35px; display: grid; place-items: center; border-radius: 9px; font-weight: 800; }
        .brand-name { color: var(--ink); font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.05rem; }
        .hero { padding: 1.6rem 1.8rem; margin: 0 0 1.6rem; background: linear-gradient(110deg, #e4efdf, #f1f4d9 68%, #e8eee2); border: 1px solid #dce7d8; border-radius: 8px; }
        .hero h1 { color: var(--ink) !important; }
        .hero p { color: #51635a; max-width: 690px; margin: .5rem 0 0; }
        [data-testid="stSidebar"] { background: #edf1ea; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        .stButton > button { border-radius: 6px; font-weight: 600; border-color: #b9c8bd; }
        .stButton > button[kind="primary"] { background: var(--green); border-color: var(--green); }
        [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); background: #fff; border-radius: 7px; }
        .match-badge { color: #165b46; background: #e1f2d2; padding: .28rem .48rem; border-radius: 4px; font-size: .67rem; font-weight: 800; }
        .price-line { font-size: 1.35rem; font-weight: 800; margin: .15rem 0 .85rem; }
        .old-price { color: #87928b; font-size: .88rem; font-weight: 400; text-decoration: line-through; margin-left: .35rem; }
        .discount { color: #1d7456; font-size: .77rem; font-weight: 700; margin-left: .35rem; }
        .spec-label { color: #75827a; font-size: .65rem; font-weight: 700; letter-spacing: .04em; }
        .spec-value { color: #27352e; font-size: .9rem; font-weight: 600; margin-top: .2rem; }
        .reason { background: #f1f6ed; border-left: 3px solid #78a558; padding: .65rem .8rem; color: #3e5947; font-size: .88rem; margin: .6rem 0; }
        [data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 7px; padding: .8rem 1rem; }
        [data-testid="stMetricLabel"] p { color: var(--muted); }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 7px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='masthead'><div class='brand-mark'>cg</div><div class='brand-name'>common ground</div></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero'><div class='eyebrow'>Your shopping desk · India</div>"
        "<h1>Find the right laptop.</h1>"
        "<p>Tell us what matters. Our product, inventory and budget agents will search the sample catalog and compare the options that actually fit.</p></div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("## Shopping brief")
    st.sidebar.caption("Describe the job. We will check the catalog against your requirements.")
    with st.sidebar.form("shopping-request"):
        request = st.text_area(
            "What are you looking for?",
            value="I need a laptop under ₹60,000 for Python programming, machine learning basics, and online classes. I prefer 16 GB RAM and good battery life.",
            height=150,
            help="Mention a budget and specs such as RAM or battery preference.",
        )
        submitted = st.form_submit_button("Find matching laptops", type="primary", use_container_width=True)
    if submitted:
        st.session_state.workflow_request = request.strip()
    st.sidebar.divider()
    st.sidebar.caption("Sample catalog · prices and stock are illustrative, not live listings.")
    _render_cart()

    catalog_tab, faq_tab, banking_tab = st.tabs(
        ["Laptop finder", "Product FAQ", "Banking support demo"]
    )
    with catalog_tab:
        if not request.strip():
            st.info("Add a shopping brief to get started.")
        elif submitted or st.session_state.get("workflow_request") == request.strip():
            _render_results(st.session_state.get("workflow_request", request.strip()))
        else:
            st.markdown("### Ready when you are")
            st.write("Your requirements stay in the brief. Select **Find matching laptops** to run the shopping agents.")
    with faq_tab:
        st.markdown("### Ask about products")
        st.caption("Answers are retrieved from the local product FAQ. Add an OpenAI key to have the assistant phrase an answer from those sources.")
        question = st.text_input("Your question", placeholder="What does the warranty cover?")
        if st.button("Search product FAQ", key="faq-search") and question.strip():
            with st.spinner("Searching product guidance..."):
                try:
                    answer, sources = retrieve_faq_answer(question.strip())
                    st.markdown(f"**Answer**\n\n{answer}")
                    if sources:
                        with st.expander("Retrieved FAQ sources"):
                            for source in sources:
                                st.write(source)
                except Exception as error:
                    st.error(f"The FAQ search could not complete: {error}")
    with banking_tab:
        _render_banking_assistant()


if __name__ == "__main__":
    main()