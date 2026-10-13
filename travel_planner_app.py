"""Streamlit entry point for the CrewAI smart travel planning demo."""

from __future__ import annotations

import os
from datetime import date, timedelta

import streamlit as st
from dotenv import load_dotenv

from travel_planner import TravelPlannerError, TravelRequest, generate_travel_plan

load_dotenv()

st.set_page_config(page_title="Smart Travel Planner", page_icon="🧭", layout="wide")
st.title("🧭 Smart Travel Planning Crew")
st.caption("A CrewAI multi-agent travel-planning training demo")
st.info(
    "Travel research and weather use public APIs (or Tavily when configured). Hotel and transport "
    "prices are mock estimates only; this app does not book travel."
)

with st.form("travel_request"):
    first, second = st.columns(2)
    with first:
        destination = st.text_input("Destination", placeholder="e.g. Kochi")
        starting_location = st.text_input("Starting location", placeholder="e.g. Bengaluru")
        start_date = st.date_input("Travel start date", value=date.today() + timedelta(days=7))
        end_date = st.date_input("Travel end date", value=date.today() + timedelta(days=10))
    with second:
        budget_inr = st.number_input("Total budget (₹)", min_value=1_000, value=30_000, step=1_000)
        interests = st.multiselect(
            "Preferred activities",
            ["Sightseeing", "Shopping", "Adventure", "Relaxation", "Food", "Culture", "Nature"],
            default=["Sightseeing", "Food"],
        )
        provider = st.selectbox("LLM provider", ["Ollama", "OpenRouter"])
        provider_key = provider.lower()
        default_model = (
            os.getenv("OLLAMA_MODEL", "ollama/llama3.1")
            if provider_key == "ollama"
            else os.getenv("OPENROUTER_MODEL", "openrouter/openai/gpt-4o-mini")
        )
        model = st.text_input("Model", value=default_model)
    submitted = st.form_submit_button("Plan my trip", type="primary")

if submitted:
    request = TravelRequest(
        destination=destination.strip(),
        starting_location=starting_location.strip(),
        start_date=start_date,
        end_date=end_date,
        budget_inr=float(budget_inr),
        interests=tuple(interests),
    )
    try:
        with st.spinner("Researching your trip and coordinating the CrewAI agents…"):
            result = generate_travel_plan(request, provider_key, model.strip() or None)
    except TravelPlannerError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(f"The travel request or provider configuration is invalid: {exc}")
    else:
        st.subheader("Coordinated travel plan")
        st.markdown(result["plan"])

        st.subheader("Budget check")
        budget = result["budget"]
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Estimated total", f"₹{budget['total_estimate_inr']:,.0f}")
        metric_b.metric("Your budget", f"₹{budget['budget_inr']:,.0f}")
        metric_c.metric(
            "Budget status",
            "Within budget" if budget["within_budget"] else "Over budget",
            f"₹{abs(budget['remaining_inr']):,.0f} "
            f"{'remaining' if budget['within_budget'] else 'over'}",
        )
        st.json(budget["breakdown_inr"])

        if result["revision_triggered"]:
            st.warning("The first estimate exceeded your budget; the agents ran a lower-cost revision.")
        if result["alternatives"]:
            st.subheader("Budget alternatives")
            for alternative in result["alternatives"]:
                st.write(f"- {alternative}")

        if result["weather"].get("available"):
            with st.expander("Weather forecast"):
                st.json(result["weather"])
        else:
            st.warning(result["weather"].get("message") or result["weather"].get("error") or "Forecast unavailable.")
        with st.expander("Research, route, and selected mock options"):
            st.write("Destination research")
            st.json(result["research"])
            st.write("Route estimate")
            st.json(result["route"])
            st.write("Hotel and transport options")
            st.json(result["options"])
        if result["sources"]:
            st.subheader("Sources")
            for source in result["sources"]:
                st.markdown(f"- [{source}]({source})")
        for warning in result["warnings"]:
            st.warning(warning)

st.divider()
st.caption(
    "Prototype only. Forecasts can change; prices are not live quotes; verify information directly "
    "with providers before travel. No reservation or payment is made."
)
