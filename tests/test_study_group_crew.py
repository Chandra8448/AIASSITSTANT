from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from study_group_crew import (
    StudyGroupError,
    _adapt_tool,
    _build_crew,
    _final_output,
    generate_study_pack,
    openalex_search,
    wikipedia_search,
)


class LangChainToolTests(unittest.TestCase):
    def test_crewai_adapter_maps_query_to_langchain_input_name(self) -> None:
        tool = Mock()
        tool.name = "sample_search"
        tool.description = "A sample search tool."
        tool.args = {"query": {"type": "string"}}
        tool.invoke.return_value = "sample result"

        adapter = _adapt_tool(tool)

        self.assertEqual(adapter.run(query="photosynthesis"), "sample result")
        tool.invoke.assert_called_once_with({"query": "photosynthesis"})

    @patch("study_group_crew.requests.get")
    def test_wikipedia_langchain_tool_returns_citations(self, get: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "query": {
                "search": [
                    {"title": "Photosynthesis", "snippet": "Light energy is converted."}
                ]
            }
        }
        get.return_value = response

        result = wikipedia_search.invoke({"query": "photosynthesis"})

        self.assertIn("Photosynthesis", result)
        self.assertIn("https://en.wikipedia.org/wiki/Photosynthesis", result)

    @patch("study_group_crew.requests.get")
    def test_openalex_langchain_tool_includes_publication_links(self, get: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "title": "Photosynthesis research",
                    "publication_year": 2024,
                    "doi": "https://doi.org/10.1234/example",
                    "primary_location": {"source": {"display_name": "Example Journal"}},
                }
            ]
        }
        get.return_value = response

        result = openalex_search.invoke({"query": "photosynthesis"})

        self.assertIn("https://doi.org/10.1234/example", result)
        self.assertIn("Example Journal", result)

    @patch(
        "study_group_crew.requests.get",
        side_effect=requests.ConnectionError("offline"),
    )
    def test_search_tool_reports_network_failure(self, get: Mock) -> None:
        result = wikipedia_search.invoke({"query": "photosynthesis"})

        self.assertIn("Research request failed", result)
        self.assertIn("offline", result)
        get.assert_called_once()


class CrewTests(unittest.TestCase):
    def test_four_agents_sequential_tasks_and_explicit_context(self) -> None:
        with (
            patch("study_group_crew._create_llm", return_value=object()),
            patch("study_group_crew.Agent") as agent_factory,
            patch("study_group_crew.Task") as task_factory,
            patch("study_group_crew.Crew") as crew_factory,
        ):
            task_results = [object() for _ in range(4)]
            task_factory.side_effect = task_results
            crew, review_task = _build_crew("Photosynthesis", "ollama", None)

        self.assertEqual(agent_factory.call_count, 4)
        self.assertEqual(task_factory.call_count, 4)
        researcher_tools = agent_factory.call_args_list[0].kwargs["tools"]
        checker_tools = agent_factory.call_args_list[3].kwargs["tools"]
        self.assertEqual({tool.name for tool in researcher_tools}, {"wikipedia_search", "openalex_search"})
        self.assertEqual({tool.name for tool in checker_tools}, {"wikipedia_search", "openalex_search"})
        args = crew_factory.call_args.kwargs
        self.assertEqual(len(args["agents"]), 4)
        self.assertEqual(len(args["tasks"]), 4)
        self.assertEqual(str(args["process"]), "Process.sequential")
        task_calls = task_factory.call_args_list
        self.assertEqual(task_calls[0].kwargs["agent"], agent_factory.return_value)
        self.assertIsNone(task_calls[0].kwargs.get("context"))
        self.assertEqual(task_calls[1].kwargs["context"], [task_results[0]])
        self.assertEqual(task_calls[2].kwargs["context"], task_results[:2])
        self.assertEqual(task_calls[3].kwargs["context"], task_results[:3])
        self.assertIs(review_task, task_results[3])
        self.assertIsNotNone(crew)

    def test_successful_generation_saves_validated_pack_and_reflection(self) -> None:
        valid_output = (
            "## Fact-check and corrections\nClaims checked.\n\n"
            "## FINAL STUDY PACK\n"
            + "Guide, quiz, answer key, and citations. " * 10
        )
        crew = Mock()
        crew.kickoff.return_value = SimpleNamespace(raw=valid_output)
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "study_group_output.md"
            reflection_path = Path(temp_dir) / "study_group_reflection.md"
            with (
                patch(
                    "study_group_crew._build_crew",
                    return_value=(crew, Mock(output=None)),
                ),
                patch("study_group_crew.STUDY_PACK_PATH", output_path),
                patch("study_group_crew.REFLECTION_PATH", reflection_path),
            ):
                result = generate_study_pack("Photosynthesis", "ollama", "test-model")

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                valid_output.rstrip() + "\n",
            )
            self.assertIn("What worked well", reflection_path.read_text(encoding="utf-8"))
            self.assertEqual(result["output_path"], output_path)

    def test_final_output_requires_final_pack_section(self) -> None:
        with self.assertRaisesRegex(StudyGroupError, "FINAL STUDY PACK"):
            _final_output("No final output.", Mock(output=None))

    def test_final_output_keeps_fact_check_and_pack(self) -> None:
        text = (
            "## Fact-check and corrections\nVerified.\n\n"
            "## FINAL STUDY PACK\n"
            + "A cited study pack with a guide, quiz, answer key, and sources. " * 8
        )
        result = _final_output(Mock(raw=text), Mock(output=None))

        self.assertTrue(result.startswith("## Fact-check"))
        self.assertIn("## FINAL STUDY PACK", result)


if __name__ == "__main__":
    unittest.main()
