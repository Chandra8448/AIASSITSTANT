"""CrewAI study group that researches a topic and creates a fact-checked study pack."""

from __future__ import annotations

import argparse
import json
import os
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
STUDY_PACK_PATH = PROJECT_DIR / "study_group_output.md"
REFLECTION_PATH = PROJECT_DIR / "study_group_reflection.md"
REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = "ClaudePracticeStudyGroup/1.0 (educational project)"
MAX_SEARCH_RESULTS = 5


class StudyGroupError(ValueError):
    """Raised when research, model generation, or output validation fails."""


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise StudyGroupError(f"Research request failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise StudyGroupError("Research service returned an unexpected response.")
    return payload


@langchain_tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for an overview of the requested study topic."""
    try:
        payload = _get_json(
            "https://en.wikipedia.org/w/api.php",
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": MAX_SEARCH_RESULTS,
                "format": "json",
            },
        )
    except StudyGroupError as exc:
        return json.dumps({"tool": "Wikipedia search", "error": str(exc)})

    results = payload.get("query", {}).get("search", [])
    pages: list[dict[str, str]] = []
    for result in results:
        title = str(result.get("title", ""))
        if title:
            pages.append(
                {
                    "title": title,
                    "snippet": str(result.get("snippet", "")),
                    "url": "https://en.wikipedia.org/wiki/"
                    + requests.utils.quote(title.replace(" ", "_"), safe="()"),
                }
            )
    return json.dumps(
        {"tool": "Wikipedia search", "query": query, "results": pages},
        ensure_ascii=False,
    )


@langchain_tool
def openalex_search(query: str) -> str:
    """Find scholarly works about the requested topic using OpenAlex."""
    try:
        payload = _get_json(
            "https://api.openalex.org/works",
            {"search": query, "per-page": MAX_SEARCH_RESULTS},
        )
    except StudyGroupError as exc:
        return json.dumps({"tool": "OpenAlex scholarly search", "error": str(exc)})

    results: list[dict[str, str]] = []
    for work in payload.get("results", []):
        title = str(work.get("title") or "")
        if not title:
            continue
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        results.append(
            {
                "title": title,
                "year": str(work.get("publication_year") or "unknown"),
                "doi": str(work.get("doi") or ""),
                "url": str(work.get("doi") or work.get("id") or ""),
                "journal": str(source.get("display_name") or ""),
            }
        )
    return json.dumps(
        {"tool": "OpenAlex scholarly search", "query": query, "results": results},
        ensure_ascii=False,
    )


class LangChainToolAdapter(BaseTool):
    """Adapt a single-input LangChain tool to CrewAI's BaseTool interface."""

    name: str
    description: str
    lc_tool: Any

    def _run(self, query: str) -> str:
        arguments = getattr(self.lc_tool, "args", {})
        if not isinstance(arguments, dict) or len(arguments) != 1:
            raise StudyGroupError(
                f"LangChain tool {self.name!r} must have exactly one input field."
            )
        argument_name = next(iter(arguments))
        return str(self.lc_tool.invoke({argument_name: query}))


def _adapt_tool(lc_tool: Any) -> LangChainToolAdapter:
    return LangChainToolAdapter(
        name=lc_tool.name,
        description=lc_tool.description,
        lc_tool=lc_tool,
    )


def _create_llm(provider: str | None = None, model: str | None = None) -> LLM:
    selected_provider = (provider or os.getenv("STUDY_LLM_PROVIDER", "ollama")).lower()
    if selected_provider == "ollama":
        return LLM(
            model=model or os.getenv("OLLAMA_MODEL", "ollama/llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.2,
        )
    if selected_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise StudyGroupError("Set OPENAI_API_KEY to use the OpenAI provider.")
        return LLM(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=0.2,
        )
    if selected_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise StudyGroupError("Set OPENROUTER_API_KEY to use OpenRouter.")
        return LLM(
            model=model or os.getenv("OPENROUTER_MODEL", "openrouter/openai/gpt-4o-mini"),
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
        )
    raise StudyGroupError("Choose Ollama, OpenAI, or OpenRouter as the LLM provider.")


def _build_crew(topic: str, provider: str | None, model: str | None) -> tuple[Crew, Task]:
    llm = _create_llm(provider, model)
    research_tools = [_adapt_tool(wikipedia_search), _adapt_tool(openalex_search)]

    researcher = Agent(
        role="Topic Researcher",
        goal="Gather accurate, relevant, cited facts about the learner's topic.",
        backstory=(
            "You are a careful academic librarian. You search authoritative and scholarly "
            "sources, record URLs, and clearly distinguish evidence from uncertainty."
        ),
        tools=research_tools,
        llm=llm,
        max_iter=3,
        max_execution_time=120,
        verbose=False,
    )
    explainer = Agent(
        role="Explainer",
        goal="Teach the topic clearly at an introductory student level without changing its meaning.",
        backstory=(
            "You are a patient tutor who explains difficult ideas in plain language, "
            "defines essential terms, and ties every factual claim to research."
        ),
        llm=llm,
        max_iter=3,
        max_execution_time=120,
        verbose=False,
    )
    quiz_maker = Agent(
        role="Quiz Maker",
        goal="Create a balanced quiz and answer key that tests understanding of the researched topic.",
        backstory=(
            "You are an experienced educator who writes unambiguous questions, "
            "varies question types, and grounds every answer in the supplied study guide."
        ),
        llm=llm,
        max_iter=3,
        max_execution_time=120,
        verbose=False,
    )
    fact_checker = Agent(
        role="Fact-Checker",
        goal="Verify the study guide and quiz against source findings, correct errors, and produce the final study pack.",
        backstory=(
            "You are a skeptical subject editor. You reject unsupported claims, check "
            "answer keys against evidence, preserve source links, and state unresolved gaps."
        ),
        tools=research_tools,
        llm=llm,
        max_iter=3,
        max_execution_time=120,
        verbose=False,
    )

    research_task = Task(
        description=(
            f"Research the topic: {topic!r}. Use BOTH wikipedia_search and "
            "openalex_search tools. Collect 5-8 accurate facts, important vocabulary, "
            "at least two useful source URLs when available, and source limitations. "
            "Never make up sources or fill gaps with guesses. Clearly report tool errors."
        ),
        expected_output=(
            "A concise evidence sheet with claims, source URLs, relevance, and limitations."
        ),
        agent=researcher,
    )
    explain_task = Task(
        description=(
            f"Create an accurate beginner-friendly study guide for {topic!r} from the "
            "research task only. Explain key concepts and terms in clear language, "
            "preserve source links, and mark any unresolved uncertainty."
        ),
        expected_output="A structured study guide with sourced claims and a short summary.",
        agent=explainer,
        context=[research_task],
    )
    quiz_task = Task(
        description=(
            "Create a five-question quiz based only on the study guide and research. "
            "Include a mix of multiple-choice and short-answer questions, followed by "
            "an answer key with brief explanations. Do not test facts absent from context."
        ),
        expected_output="Five clear questions and a separate, justified answer key.",
        agent=quiz_maker,
        context=[research_task, explain_task],
    )
    review_task = Task(
        description=(
            f"Fact-check the study guide and quiz for {topic!r}. Use BOTH search tools "
            "to verify key factual claims where possible. Compare claims and answers "
            "with the source evidence, remove or qualify anything unsupported, ensure "
            "citations are present, and preserve tool failures as explicit limitations. "
            "Return exactly these headings: '## Fact-check and corrections', then "
            "'## FINAL STUDY PACK' containing a concise guide, five-question quiz, "
            "answer key, and sources. Do not invent findings."
        ),
        expected_output=(
            "A short fact-check note followed by a corrected, fully cited study pack."
        ),
        agent=fact_checker,
        context=[research_task, explain_task, quiz_task],
    )
    crew = Crew(
        agents=[researcher, explainer, quiz_maker, fact_checker],
        tasks=[research_task, explain_task, quiz_task, review_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew, review_task


def _final_output(result: Any, review_task: Task) -> str:
    task_output = getattr(review_task, "output", None)
    raw = getattr(task_output, "raw", None)
    if not isinstance(raw, str) or not raw.strip():
        raw = getattr(result, "raw", None)
    if not isinstance(raw, str) or not raw.strip():
        raw = str(result).strip()
    if "## FINAL STUDY PACK" not in raw:
        raise StudyGroupError(
            "The fact-checker did not return the required FINAL STUDY PACK section."
        )
    final_pack = raw[raw.index("## FINAL STUDY PACK") :].strip()
    if len(final_pack) < 200:
        raise StudyGroupError("The final study pack is incomplete.")
    return raw.strip() + "\n"


def _write_reflection(topic: str) -> None:
    reflection = f"""# Study Group Crew reflection

This project separates learning into research, explanation, quiz creation, and independent fact-checking. The Topic Researcher uses two LangChain tools—Wikipedia search for an accessible overview and OpenAlex for scholarly works—and passes its evidence sheet to the Explainer. The Quiz Maker receives both the research and study guide, so questions and answers can be traced to the material. The Fact-Checker reviews the entire chain, checks key claims with the research tools, and writes the final study pack to `study_group_output.md`.

What worked well is the explicit sequential hand-off: each task has a focused purpose and receives the prior outputs as context. Tool results include source links, and the final reviewer is instructed to remove unsupported statements instead of filling evidence gaps with guesses. The focused tests and full project suite pass, and CrewAI constructs all four agents and tasks.

The live end-to-end run with the local Ollama model did not finish: CrewAI returned a `TimeoutError` before the final study pack was produced. The checked-in pack is therefore a prepared example, not output from that live run. Public search can also fail or return weak sources, and an LLM may overlook a contradiction or produce an incomplete answer key. Tool errors are reported in tool results rather than hidden, but successful search alone does not prove a claim is true.

Next, I would profile and shorten local-model prompts, then rerun the end-to-end workflow and confirm fresh output timestamps. I would evaluate claim-level citation support and quiz-answer accuracy against instructor-reviewed examples, add human review for uncertain claims, and compare sequential with hierarchical orchestration. Try the system with the topic {topic!r}; independently verify cited sources before using generated material for an exam.
"""
    REFLECTION_PATH.write_text(reflection, encoding="utf-8")


def generate_study_pack(
    topic: str,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Run the sequential study group and save the reviewed study pack and reflection."""
    load_dotenv()
    topic = topic.strip()
    if not topic:
        raise StudyGroupError("Enter a topic to research.")
    crew, review_task = _build_crew(topic, provider, model)
    try:
        result = crew.kickoff()
    except (APIError, OpenAIAPIError, ConnectionError, TimeoutError) as exc:
        raise StudyGroupError(
            f"The selected LLM could not complete the crew ({type(exc).__name__}). "
            "Check Ollama and the selected model, or verify the configured provider key."
        ) from exc
    output = _final_output(result, review_task)
    STUDY_PACK_PATH.write_text(output, encoding="utf-8")
    _write_reflection(topic)
    return {
        "study_pack": output,
        "output_path": STUDY_PACK_PATH,
        "reflection_path": REFLECTION_PATH,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research a topic and create a fact-checked study guide and quiz."
    )
    parser.add_argument("topic", nargs="?", default="Photosynthesis")
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai", "openrouter"),
        default=None,
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    try:
        result = generate_study_pack(args.topic, args.provider, args.model)
    except StudyGroupError as exc:
        raise SystemExit(f"Study Group generation failed: {exc}") from exc
    print(f"Study pack saved to {result['output_path']}")
    print(f"Reflection saved to {result['reflection_path']}")


if __name__ == "__main__":
    main()
