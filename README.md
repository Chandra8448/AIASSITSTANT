# Common Ground: Shopping and Banking Assistant

A Streamlit demo with a laptop-shopping assistant and a smart banking support agent. The shopping assistant uses LangChain tools, a LangGraph workflow, a local CSV catalog, and Chroma FAQ retrieval. The banking assistant uses a single LangGraph tool-calling agent with LangChain tools and a local SQLite database of fictional sample customers.

All product and banking information is illustrative. The cart is simulated. The banking eligibility check is a sample affordability estimate, not a bank decision; submitted applications are saved locally with a `submitted_for_review` status and are not loan approvals.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

The app works without an OpenAI API key: shopping decisions use the local catalog and deterministic scoring, FAQ answers use local Chroma retrieval, and banking uses its deterministic local demo flow. Add `OPENAI_API_KEY` to `.env` to enable OpenAI-written shopping explanations and model-directed banking tool selection. Set `OPENAI_MODEL` to select a compatible chat model.

## Banking assistant

Run the dedicated banking-only experience with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run banking_app.py
```

This entry point has no shopping brief, cart, or product tabs. The original combined shopping app still has a **Banking support demo** tab if you want the combined experience. Select one of the fictional sample customers in either banking experience. The prefilled example requests ₹5,00,000 over three years; the agent uses a 12% illustrative annual rate, retrieves the masked sample account, calculates the EMI, and checks affordability against the sample salary and existing EMI. The sample eligibility rule limits total monthly debt to 50% of the sample salary.

The agent can calculate, retrieve, and assess, but it cannot authorize its own application submission. The app displays the application details and requires the customer to press **Confirm and submit loan application**. The backend then creates a short-lived, one-use approval code and rechecks eligibility before saving the application. Only a masked account number is displayed. The local SQLite database is created at `data/banking_demo.sqlite3` on first banking use and is ignored by Git.

The module's tool and confirmation flow tests can be run with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## CrewAI travel planner

Run the dedicated multi-agent travel experience with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run travel_planner_app.py
```

The sequential CrewAI crew has a Travel Coordinator, Destination Research Agent,
Hotel and Transport Agent, Budget Analyst, and Itinerary Planner. Task outputs
are passed as context to downstream tasks. Public search (Wikipedia by default,
or Tavily when `TAVILY_API_KEY` is configured), Open-Meteo weather/geocoding,
and OSRM route tools provide external research. Weather is explicitly marked
unavailable when travel dates exceed the forecast window. Hotel, intercity
transport, local travel, meal, and activity prices are labeled mock estimates;
no booking API is connected and nothing is reserved.

Choose Ollama, OpenRouter, or OpenAI in the app. Ollama needs a running local
Ollama server and a locally installed model. OpenRouter requires
`OPENROUTER_API_KEY`; OpenAI requires `OPENAI_API_KEY`. For hosted deployments,
add the matching key as an environment variable in the hosting provider and
select that same provider in the app.
Provider/model settings are in `.env.example`. If the initial standard estimate
exceeds the budget, the planner runs a second CrewAI pass using lower-cost mock
options and reports whether the revised estimate still exceeds the budget.

## Deploy to Render

The included `render.yaml` defines a free Python web service, Streamlit's Render port binding, and a health check. To deploy it:

1. Push this project to a GitHub, GitLab, or Bitbucket repository.
2. In the [Render Dashboard](https://dashboard.render.com), choose **New +** > **Blueprint** and connect that repository.
3. Confirm `render.yaml` and apply the Blueprint. Render builds the app and provides its public `onrender.com` URL.
4. To enable OpenAI-written explanations, add `OPENAI_API_KEY` under the service's **Environment** settings and redeploy. The app runs without it.

The sample catalog and FAQ CSVs are included in the deployment. Render's local filesystem is ephemeral, so Chroma rebuilds its small FAQ index from the CSV when needed. The cart is simulated and isn't shared across visitors or service restarts.

## Project layout

- `app.py`: Streamlit UI, LangChain tools, LangGraph agents, and Chroma FAQ retrieval.
- `banking_app.py`: standalone banking-only Streamlit UI.
- `banking_agent.py`: sample banking data service, LangChain tools, and the LangGraph banking agent.
- `travel_planner_app.py`: dedicated Streamlit UI for the CrewAI travel planner.
- `travel_planner.py`: travel request validation, external/mock tools, CrewAI agents/tasks, and budget revision.
- `tests/test_banking_agent.py`: banking EMI, eligibility, and confirmation-gate tests.
- `tests/test_travel_planner.py`: travel validation, cost calculation, and mock/forecast tests.
- `data/product_catalog.csv`: sample products and inventory.
- `data/product_faq.csv`: source material for FAQ retrieval.
- `data/banking_demo.sqlite3`: generated fictional banking demo data and submitted applications.
- `.chroma/`: local Chroma database created on first FAQ search.