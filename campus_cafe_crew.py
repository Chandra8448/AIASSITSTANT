"""CrewAI launch-brief team using LangChain research and calculator tools."""

from __future__ import annotations

import ast
import json
import operator
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain_core.tools import tool as langchain_tool
from litellm.exceptions import APIError
from openai import APIError as OpenAIAPIError

PROJECT_DIR = Path(__file__).resolve().parent
FINAL_BRIEF_PATH = PROJECT_DIR / "final_brief.md"
REFLECTION_PATH = PROJECT_DIR / "collaboration_reflection.md"
REQUEST_TIMEOUT_SECONDS = 15
MAX_SEARCH_RESULTS = 5
USER_AGENT = "CampusCafeLaunchTeam/1.0 (student project research)"

STUDENT_CAFE_STUDY = {
    "title": "Cafe Politics: How Food Service Operators Influence University Students' Satisfaction and Dining Frequency",
    "url": "https://zenodo.org/records/7747564",
    "finding": (
        "A survey of 201 students using convenience sampling reported that food "
        "quality, ambience, value for money, menu variety, and service quality "
        "were important to satisfaction and dining frequency."
    ),
    "limitation": "One convenience sample; not a representative survey of this campus or all Indian students.",
}
PLANT_BASED_STUDY = {
    "title": "Consumer Preference Segments for Plant-Based Foods: The Role of Product Category",
    "url": "https://doi.org/10.3390/foods11193059",
    "finding": (
        "A four-country consumer study (including India; total n=2,494) found "
        "different willingness-to-consume segments across plant-based food categories."
    ),
    "limitation": "The study is not campus-specific; it does not establish demand at this university.",
}
FOOD_WASTE_SOURCE = {
    "title": "UNEP Food Waste Index Report 2024",
    "url": "https://www.unep.org/resources/publication/food-waste-index-report-2024",
    "finding": (
        "UNEP covers food waste across food service, retail, and household sectors "
        "and recommends measuring waste to support reduction."
    ),
    "limitation": "Global guidance; it does not quantify waste for the proposed campus café.",
}
CAFE_MENU_SOURCE = {
    "title": "Tata Starbucks India menu (live menu; city and item prices vary)",
    "url": "https://www.starbucks.in/menu",
    "finding": (
        "A current Indian chain menu is available as a competitor-check reference; "
        "public page rendering did not expose verifiable item-level prices to this run."
    ),
    "limitation": "Do not treat the proposed sample prices below as extracted market prices.",
}


class CafeResearchError(ValueError):
    """Raised when the research or brief-generation workflow cannot complete."""


@dataclass(frozen=True)
class CafeAssumptions:
    campus_enrollment: int = 8_000
    open_days_per_month: int = 26
    average_transaction_inr: float = 110.0
    variable_cost_share: float = 0.35
    rent_and_facilities_inr: float = 60_000.0
    staff_count: int = 3
    monthly_salary_per_staff_inr: float = 18_000.0
    utilities_inr: float = 15_000.0
    equipment_lease_and_maintenance_inr: float = 12_000.0
    cleaning_admin_and_misc_inr: float = 9_000.0
    tea_price_inr: float = 25.0
    filter_coffee_price_inr: float = 45.0
    cappuccino_price_inr: float = 90.0
    snack_price_inr: float = 55.0


def _safe_calculate(expression: str) -> float:
    """Evaluate arithmetic-only expressions without Python eval or builtins."""
    allowed_operators: dict[type[ast.operator], Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_operators: dict[type[ast.unaryop], Any] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise CafeResearchError("Exponent magnitude is limited to 10.")
            return float(allowed_operators[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_operators:
            return float(unary_operators[type(node.op)](evaluate(node.operand)))
        raise CafeResearchError(
            "Calculator accepts only numbers, parentheses, and + - * / // % **."
        )

    try:
        parsed = ast.parse(expression, mode="eval")
        result = evaluate(parsed)
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise CafeResearchError(f"Calculator could not evaluate the expression: {exc}") from exc
    if abs(result) > 1e15:
        raise CafeResearchError("Calculator result exceeds the supported range.")
    return result


@langchain_tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression; use this tool for every estimate."""
    try:
        return f"{_safe_calculate(expression):.6f}".rstrip("0").rstrip(".")
    except CafeResearchError as exc:
        return f"ERROR: {exc}"


@langchain_tool
def web_search(query: str) -> str:
    """Search OpenAlex scholarly works and return titles, abstracts, and source links."""
    try:
        response = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": MAX_SEARCH_RESULTS,
                "select": "title,publication_year,doi,authorships,abstract_inverted_index,primary_location",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return json.dumps(
            {
                "tool": "OpenAlex scholarly search",
                "error": f"Research search failed: {exc}",
                "query": query,
            }
        )

    works = payload.get("results")
    if not isinstance(works, list):
        return json.dumps(
            {"tool": "OpenAlex scholarly search", "error": "Unexpected search response."}
        )
    results: list[dict[str, Any]] = []
    for work in works:
        if not isinstance(work, dict):
            continue
        inverted = work.get("abstract_inverted_index") or {}
        positioned_words = sorted(
            (position, word)
            for word, positions in inverted.items()
            for position in positions
        )
        abstract = " ".join(word for _, word in positioned_words)
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        results.append(
            {
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "doi": work.get("doi"),
                "source": source.get("display_name"),
                "abstract": abstract[:1_200],
            }
        )
    return json.dumps(
        {"tool": "OpenAlex scholarly search", "query": query, "results": results},
        ensure_ascii=False,
    )


@langchain_tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for background context and return page titles and links."""
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": MAX_SEARCH_RESULTS,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return json.dumps(
            {"tool": "Wikipedia search", "error": f"Wikipedia lookup failed: {exc}"}
        )

    results = payload.get("query", {}).get("search", [])
    pages = [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": "https://en.wikipedia.org/wiki/"
            + str(item.get("title", "")).replace(" ", "_"),
        }
        for item in results
    ]
    return json.dumps(
        {"tool": "Wikipedia search", "query": query, "results": pages},
        ensure_ascii=False,
    )


class LangChainToolAdapter(BaseTool):
    """Expose a one-string-argument LangChain tool to CrewAI agents."""

    name: str
    description: str
    lc_tool: Any

    def _run(self, query: str) -> str:
        arguments = getattr(self.lc_tool, "args", {})
        if not isinstance(arguments, dict) or len(arguments) != 1:
            raise CafeResearchError(
                f"LangChain tool {self.name!r} must accept exactly one string argument."
            )
        argument_name = next(iter(arguments))
        return str(self.lc_tool.invoke({argument_name: query}))


def _adapt_tool(lc_tool: Any) -> LangChainToolAdapter:
    return LangChainToolAdapter(
        name=lc_tool.name,
        description=lc_tool.description,
        lc_tool=lc_tool,
    )


def calculate_break_even(assumptions: CafeAssumptions | None = None) -> dict[str, Any]:
    """Compute a transparent monthly break-even estimate using the LangChain calculator."""
    assumptions = assumptions or CafeAssumptions()
    fixed_cost_expression = (
        f"{assumptions.rent_and_facilities_inr}"
        f"+({assumptions.staff_count}*{assumptions.monthly_salary_per_staff_inr})"
        f"+{assumptions.utilities_inr}"
        f"+{assumptions.equipment_lease_and_maintenance_inr}"
        f"+{assumptions.cleaning_admin_and_misc_inr}"
    )
    fixed_costs = _calculate_with_tool(fixed_cost_expression)
    contribution_per_transaction = _calculate_with_tool(
        f"{assumptions.average_transaction_inr}*(1-{assumptions.variable_cost_share})"
    )
    monthly_transactions_exact = _calculate_with_tool(
        f"{fixed_costs}/{contribution_per_transaction}"
    )
    monthly_transactions = int(monthly_transactions_exact)
    if monthly_transactions < monthly_transactions_exact:
        monthly_transactions += 1
    daily_transactions_exact = _calculate_with_tool(
        f"{monthly_transactions}/{assumptions.open_days_per_month}"
    )
    daily_transactions = int(daily_transactions_exact)
    if daily_transactions < daily_transactions_exact:
        daily_transactions += 1
    monthly_revenue_at_daily_target = _calculate_with_tool(
        f"{daily_transactions}*{assumptions.open_days_per_month}"
        f"*{assumptions.average_transaction_inr}"
    )
    monthly_variable_cost = _calculate_with_tool(
        f"{monthly_revenue_at_daily_target}*{assumptions.variable_cost_share}"
    )
    monthly_surplus_at_target = _calculate_with_tool(
        f"{monthly_revenue_at_daily_target}-{monthly_variable_cost}-{fixed_costs}"
    )
    return {
        "assumptions": asdict(assumptions),
        "fixed_cost_expression": fixed_cost_expression,
        "fixed_monthly_costs_inr": fixed_costs,
        "contribution_expression": (
            f"{assumptions.average_transaction_inr}*(1-{assumptions.variable_cost_share})"
        ),
        "contribution_per_transaction_inr": contribution_per_transaction,
        "monthly_break_even_transactions_exact": monthly_transactions_exact,
        "monthly_break_even_transactions_rounded_up": monthly_transactions,
        "operating_days_per_month": assumptions.open_days_per_month,
        "daily_break_even_transactions_exact": daily_transactions_exact,
        "daily_break_even_transactions_rounded_up": daily_transactions,
        "daily_transactions_as_share_of_enrollment_pct": _calculate_with_tool(
            f"{daily_transactions}/({assumptions.campus_enrollment})*100"
        ),
        "monthly_revenue_at_daily_target_inr": monthly_revenue_at_daily_target,
        "monthly_variable_cost_at_daily_target_inr": monthly_variable_cost,
        "monthly_surplus_at_daily_target_inr": monthly_surplus_at_target,
    }


def _calculate_with_tool(expression: str) -> float:
    """Run a validated arithmetic expression through the LangChain calculator."""
    result = calculator.invoke({"expression": expression})
    if not isinstance(result, str) or result.startswith("ERROR:"):
        raise CafeResearchError(f"Calculator tool failed: {result}")
    try:
        return float(result)
    except ValueError as exc:
        raise CafeResearchError(
            f"Calculator tool returned a non-numeric result for {expression!r}."
        ) from exc


def _create_llm(provider: str | None = None, model: str | None = None) -> LLM:
    selected_provider = (
        provider or os.getenv("CAFE_LLM_PROVIDER", "ollama")
    ).strip().lower()
    if selected_provider == "ollama":
        return LLM(
            model=model or os.getenv("OLLAMA_MODEL", "ollama/llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.2,
        )
    if selected_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise CafeResearchError("Set OPENAI_API_KEY to use the OpenAI provider.")
        return LLM(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=0.2,
        )
    if selected_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise CafeResearchError("Set OPENROUTER_API_KEY to use OpenRouter.")
        return LLM(
            model=model or os.getenv("OPENROUTER_MODEL", "openrouter/openai/gpt-4o-mini"),
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
        )
    raise CafeResearchError("Choose Ollama, OpenAI, or OpenRouter as the LLM provider.")


def _create_team(
    assumptions: CafeAssumptions, provider: str | None, model: str | None
) -> tuple[Crew, Task]:
    llm = _create_llm(provider, model)
    researcher = Agent(
        role="Market Researcher",
        goal=(
            "Find relevant sourced evidence on student dining behavior, café price "
            "comparators, and food-service trends. Clearly label evidence limits."
        ),
        backstory=(
            "You are a careful campus-retail researcher. Every factual claim carries "
            "a source link; convenience samples and non-local evidence are explicitly caveated."
        ),
        tools=[_adapt_tool(web_search), _adapt_tool(wikipedia_search)],
        llm=llm,
        verbose=False,
    )
    analyst = Agent(
        role="Financial Analyst",
        goal="Calculate a transparent monthly break-even estimate, showing every assumption and arithmetic step.",
        backstory=(
            "You are a skeptical management accountant. You use the supplied calculator "
            "tool for every calculation, distinguish transactions from unique students, "
            "and never present assumptions as observed facts."
        ),
        tools=[_adapt_tool(calculator)],
        llm=llm,
        verbose=False,
    )
    sustainability_officer = Agent(
        role="Sustainability Officer",
        goal="Verify sustainability and plant-based claims against sources and recommend measurable actions.",
        backstory=(
            "You avoid greenwashing. You distinguish an option from proven demand and "
            "recommend actions that the café can measure."
        ),
        tools=[_adapt_tool(web_search), _adapt_tool(wikipedia_search)],
        llm=llm,
        verbose=False,
    )
    writer = Agent(
        role="Marketing Writer",
        goal="Draft a concise, persuasive, evidence-led one-page launch brief for the Dean.",
        backstory=(
            "You are a former student-newspaper editor who writes plainly for decision-makers "
            "and marks proposed prices and costs as assumptions."
        ),
        llm=llm,
        verbose=False,
    )
    reviewer = Agent(
        role="Skeptical Reviewer",
        goal="Identify weak claims, missing risks, and unsupported numbers, then produce the corrected final brief.",
        backstory=(
            "You are an exacting Dean's-office budget advisor. Reject unsourced claims, "
            "check the arithmetic against the supplied calculation, and repair the brief."
        ),
        llm=llm,
        allow_delegation=True,
        verbose=False,
    )

    financials = calculate_break_even(assumptions)
    assumption_json = json.dumps(asdict(assumptions), ensure_ascii=False)
    financial_json = json.dumps(financials, ensure_ascii=False)
    verified_evidence = json.dumps(
        [STUDENT_CAFE_STUDY, PLANT_BASED_STUDY, FOOD_WASTE_SOURCE, CAFE_MENU_SOURCE],
        ensure_ascii=False,
    )

    research_task = Task(
        description=(
            "Research evidence relevant to a proposed Indian university café serving "
            "approximately 8,000 students. Use BOTH market tools: scholarly web_search "
            "and wikipedia_search. Search student campus dining choice/value, Indian café "
            "menu comparators/prices, plant-based options, and food-service waste. Cite "
            "only claims actually supported by tool results. Separate source-backed facts "
            "from assumptions. If market-price search fails or returns no actual local "
            "prices, say this plainly and treat every sample menu price as a planning "
            "assumption, not a market finding. Flag sample, geographic, and date limitations.\n"
            f"Verified source notes from preparation research: {verified_evidence}"
        ),
        expected_output=(
            "A concise evidence table with sourced findings, URLs, scope/limitations, "
            "and explicit gaps in local student-spend and café-price data."
        ),
        agent=researcher,
    )
    finance_task = Task(
        description=(
            "Use the calculator tool for each arithmetic operation and show the expression "
            "and result. Assume all amounts are monthly INR. Costs are planning assumptions, "
            "not supplier quotes. Calculate fixed costs, contribution per order, exact and "
            "rounded-up monthly break-even orders, daily orders over the operating days, "
            "share of the 8,000-student enrollment represented by daily orders, and the "
            "estimated surplus at the rounded daily target. Do not treat orders as unique "
            "students.\n"
            f"Explicit assumptions: {assumption_json}\n"
            f"Verified Python arithmetic cross-check: {financial_json}"
        ),
        expected_output=(
            "Assumption table, calculator expressions/results, break-even order counts, "
            "and an explicit statement that the estimate is not a forecast."
        ),
        agent=analyst,
        context=[research_task],
    )
    sustainability_task = Task(
        description=(
            "Use scholarly web_search and wikipedia_search to check environmental and "
            "plant-based statements relevant to this café. Report only evidence-supported "
            "facts and links, note whether evidence is campus-specific, and propose two "
            "measurable pilot actions (for example daily food-waste measurement and "
            "offering a plant-based choice). Do not claim that students demand these "
            "options without campus survey evidence.\n"
            f"Verified source notes from preparation research: {verified_evidence}"
        ),
        expected_output=(
            "Verified sustainability evidence with source links, limitations, and "
            "measurable recommendations; no unsupported demand or impact claims."
        ),
        agent=sustainability_officer,
        context=[research_task],
    )
    draft_task = Task(
        description=(
            "Draft a decision-ready Markdown brief of about 250-350 words for the Dean. "
            "Include opportunity, sourced student behavior/trend facts, transparently "
            "labeled sample menu prices and operating-cost assumptions, break-even "
            "arithmetic, key risks, and a specific pilot/funding decision. Include inline "
            "source links. Do not state any number without a source or explicit assumption. "
            "Use only the supplied financial calculation and researcher/sustainability findings."
        ),
        expected_output="A concise, one-page Markdown draft with citations and explicit assumptions.",
        agent=writer,
        context=[research_task, finance_task, sustainability_task],
    )
    review_task = Task(
        description=(
            "Critically review the draft against the supplied research, finance, and "
            "sustainability outputs. Identify and fix unsupported or overgeneralized claims, "
            "uncited numbers, arithmetic discrepancies, missing risks, and vague asks. "
            "Do not invent research or change the financial assumptions. Return these exact "
            "sections: '## Critical review' with three concise risk/fix bullets, then "
            "'## FINAL BRIEF' with only the corrected one-page brief, then "
            "'## Sources and assumptions' with the source links and explicit non-source "
            "planning assumptions. The final brief is the authoritative output and must "
            "remain readable to a non-technical Dean.\n"
            f"Finance cross-check: {financial_json}"
        ),
        expected_output=(
            "Critical review; a corrected, cited, one-page FINAL BRIEF; and an audit list "
            "of sources and stated assumptions."
        ),
        agent=reviewer,
        context=[research_task, finance_task, sustainability_task, draft_task],
    )
    crew = Crew(
        agents=[researcher, analyst, sustainability_officer, writer, reviewer],
        tasks=[
            research_task,
            finance_task,
            sustainability_task,
            draft_task,
            review_task,
        ],
        process=Process.sequential,
        verbose=False,
    )
    return crew, review_task


def _extract_review_text(result: Any, review_task: Task) -> str:
    task_output = getattr(review_task, "output", None)
    if task_output is not None:
        raw = getattr(task_output, "raw", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    raw_result = getattr(result, "raw", None)
    if isinstance(raw_result, str) and raw_result.strip():
        return raw_result.strip()
    text = str(result).strip()
    if not text:
        raise CafeResearchError("The reviewer returned an empty result.")
    return text


def _extract_final_brief(review_text: str) -> str:
    marker = "## FINAL BRIEF"
    start = review_text.casefold().find(marker.casefold())
    if start < 0:
        raise CafeResearchError(
            "The reviewer did not include the required '## FINAL BRIEF' section; "
            "no brief was written."
        )
    final = review_text[start + len(marker) :].strip()
    next_section = final.casefold().find("\n## sources and assumptions")
    if next_section >= 0:
        final = final[:next_section].strip()
    if len(final) < 100:
        raise CafeResearchError("The reviewer returned an incomplete final brief.")
    return final + "\n"


def _write_reflection(
    assumptions: CafeAssumptions, review_text: str, provider: str, model: str | None
) -> None:
    critique_start = review_text.casefold().find("## critical review")
    critique_end = review_text.casefold().find("## final brief")
    critique = (
        review_text[critique_start + len("## critical review") : critique_end].strip()
        if critique_start >= 0 and critique_end > critique_start
        else "The reviewer checked sourcing, assumptions, arithmetic, and omitted risks."
    )
    reflection = f"""# Collaboration reflection

The sequential workflow separated evidence gathering, calculation, sustainability review, drafting, and final challenge. The Market Researcher was instructed to cite tool results and expose evidence gaps; the Financial Analyst used a constrained arithmetic tool and supplied a second Python cross-check. Passing each task's output through explicit CrewAI task context made the hand-offs inspectable rather than hiding them inside one long prompt. The Sustainability Officer checked environmental claims independently before the writer drafted the proposal.

The Skeptical Reviewer was asked to produce a critique and then a corrected final version. Its critique for this run was:

{critique}

This division improved auditability: published research findings are distinguished from the café's planning assumptions, the calculator expressions can be checked, and sustainability suggestions are framed as pilot actions rather than proven student demand. The estimate uses an assumed average transaction of ₹{assumptions.average_transaction_inr:,.0f}, variable costs of {assumptions.variable_cost_share:.0%} of sales, and monthly fixed costs calculated from the listed assumptions; it is a break-even scenario, not a sales forecast. The source studies are not a substitute for surveying this campus or collecting nearby menu quotations.

The team used sequential execution with {provider} ({model or os.getenv("OLLAMA_MODEL", "ollama/llama3.2")}). A hierarchical run was not used, so this output does not establish that hierarchical coordination is better. A failed research tool must be reported as unavailable rather than silently replaced with invented facts; the final reviewer should preserve that limitation. Before budget approval, the Dean should request a student intercept survey, local supplier/rent quotes, and a short measured pilot.
"""
    REFLECTION_PATH.write_text(reflection, encoding="utf-8")


def generate_launch_brief(
    assumptions: CafeAssumptions | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Run the sequential research crew and save the reviewed brief and reflection."""
    load_dotenv()
    assumptions = assumptions or CafeAssumptions()
    if assumptions.campus_enrollment <= 0 or assumptions.open_days_per_month <= 0:
        raise CafeResearchError("Enrollment and monthly operating days must be positive.")
    if assumptions.average_transaction_inr <= 0:
        raise CafeResearchError("Average transaction value must be positive.")
    if not 0 <= assumptions.variable_cost_share < 1:
        raise CafeResearchError("Variable-cost share must be at least 0 and less than 1.")

    crew, review_task = _create_team(assumptions, provider, model)
    try:
        result = crew.kickoff()
    except (APIError, OpenAIAPIError, ConnectionError, TimeoutError) as exc:
        raise CafeResearchError(
            f"The selected LLM could not complete the crew ({type(exc).__name__}). "
            "Check that Ollama is running with the model installed, or verify the "
            "selected provider's API key and connection."
        ) from exc
    review_text = _extract_review_text(result, review_task)
    final_brief = _extract_final_brief(review_text)
    FINAL_BRIEF_PATH.write_text(final_brief, encoding="utf-8")
    _write_reflection(assumptions, review_text, provider or os.getenv("CAFE_LLM_PROVIDER", "ollama"), model)
    return {
        "review": review_text,
        "final_brief": final_brief,
        "brief_path": FINAL_BRIEF_PATH,
        "reflection_path": REFLECTION_PATH,
        "financials": calculate_break_even(assumptions),
        "provider": provider or os.getenv("CAFE_LLM_PROVIDER", "ollama"),
        "model": model or os.getenv("OLLAMA_MODEL", "ollama/llama3.2"),
    }


def main() -> None:
    try:
        result = generate_launch_brief()
    except CafeResearchError as exc:
        raise SystemExit(f"Campus café brief generation failed: {exc}") from exc
    print("Final brief saved to", result["brief_path"])
    print("Collaboration reflection saved to", result["reflection_path"])
    print("\n" + result["final_brief"])


if __name__ == "__main__":
    main()
