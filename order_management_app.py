"""Streamlit interface for the CrewAI e-commerce order management demo."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from order_management import OrderManagementError, handle_order_query

load_dotenv()

st.set_page_config(
    page_title="Smart Order Support",
    page_icon="📦",
    layout="wide",
)
st.title("📦 Smart E-Commerce Order Support")
st.caption("A four-agent CrewAI order-management demonstration")
st.info(
    "This training demo uses fictional orders and inventory from local CSV files. "
    "It does not connect to a real store, carrier, or payment system. No refund, "
    "replacement, or order change is made."
)

with st.form("order_support_request"):
    order_id = st.text_input("Order ID", value="ORD1025", placeholder="e.g. ORD1025")
    question = st.text_area(
        "How can we help with this order?",
        value=(
            "The expected delivery date was yesterday, but I have not received it. "
            "Can you check the status and suggest a solution?"
        ),
        height=110,
    )
    first, second = st.columns(2)
    with first:
        provider = st.selectbox(
            "LLM provider",
            ["Ollama", "OpenAI", "OpenRouter"],
            index=0,
        ).lower()
    model_defaults = {
        "ollama": os.getenv("ORDER_OLLAMA_MODEL", "ollama/llama3.2"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "openrouter": os.getenv("OPENROUTER_MODEL", "openrouter/openai/gpt-4o-mini"),
    }
    with second:
        model = st.text_input("Model", value=model_defaults[provider])
    submitted = st.form_submit_button("Check order", type="primary")

if submitted:
    try:
        with st.spinner("Customer Support, Inventory, Delivery, and Resolution agents are working…"):
            result = handle_order_query(
                order_id=order_id,
                question=question,
                provider=provider,
                model=model.strip() or None,
            )
    except OrderManagementError as exc:
        st.error(str(exc))
    else:
        st.subheader("Customer response")
        st.markdown(result["response"])

        if result["escalation_required"]:
            st.error("Human support escalation recommended.")
            for reason in result["escalation_reasons"]:
                st.write(f"- {reason}")
        else:
            st.success("No human escalation is currently required.")

        with st.expander("Verified order, inventory, and tracking evidence"):
            st.write("Order")
            st.json(result["order"] or {"found": False, "order_id": order_id})
            st.write("Inventory")
            st.json(result["inventory"])
            st.write("Tracking")
            st.json(result["tracking"])

        with st.expander("CrewAI workflow"):
            st.write(
                "Customer Support Agent → Inventory Agent → Delivery Agent → "
                "Resolution Agent"
            )
            st.caption(f"LLM: {result['llm_provider']} · {result['llm_model']}")

st.divider()
st.subheader("Try the sample orders")
samples = st.columns(3)
samples[0].markdown("**ORD1001 — Delivered**  \nDesk lamp marked delivered.")
samples[1].markdown("**ORD1025 — Delayed**  \nWireless headphones delayed in transit.")
samples[2].markdown("**ORD1030 — Out of stock**  \nStudio headphones; one alternative may be available.")
st.caption("For a missing delivery or human-support request, the Resolution Agent escalates the case.")
