# Agentic AI Shopping Assistant

A Streamlit laptop-shopping demo using LangChain tools, a LangGraph workflow, a local CSV catalog, Chroma FAQ retrieval, and an optional OpenAI model for grounded explanations.

All prices, stock counts, discounts, and delivery estimates are illustrative. The cart is simulated and never places an order.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

The app works without an OpenAI API key: shopping decisions use the local catalog and deterministic scoring, and FAQ answers use local Chroma retrieval. To enable OpenAI-written explanations, add `OPENAI_API_KEY` to `.env`. The model only phrases explanations from catalog facts; search, stock and budget decisions still come from the local tools.

## Deploy to Render

The included `render.yaml` defines a free Python web service, Streamlit's Render port binding, and a health check. To deploy it:

1. Push this project to a GitHub, GitLab, or Bitbucket repository.
2. In the [Render Dashboard](https://dashboard.render.com), choose **New +** > **Blueprint** and connect that repository.
3. Confirm `render.yaml` and apply the Blueprint. Render builds the app and provides its public `onrender.com` URL.
4. To enable OpenAI-written explanations, add `OPENAI_API_KEY` under the service's **Environment** settings and redeploy. The app runs without it.

The sample catalog and FAQ CSVs are included in the deployment. Render's local filesystem is ephemeral, so Chroma rebuilds its small FAQ index from the CSV when needed. The cart is simulated and isn't shared across visitors or service restarts.

## Project layout

- `app.py`: Streamlit UI, LangChain tools, LangGraph agents, and Chroma FAQ retrieval.
- `data/product_catalog.csv`: sample products and inventory.
- `data/product_faq.csv`: source material for FAQ retrieval.
- `.chroma/`: local Chroma database created on first FAQ search.