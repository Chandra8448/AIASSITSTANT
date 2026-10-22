# Study Group Crew reflection

This project separates learning into research, explanation, quiz creation, and
independent fact-checking. The Topic Researcher uses two LangChain tools—Wikipedia
search for an accessible overview and OpenAlex for scholarly works—and passes
its evidence sheet to the Explainer. The Quiz Maker receives both the research
and study guide, so questions and answers can be traced to the material. The
Fact-Checker reviews the entire chain, checks key claims with the research
tools, and writes the final study pack to `study_group_output.md`.

What worked well is the explicit sequential hand-off: each task has a focused
purpose and receives prior outputs as context. Tool results include source
links, and the final reviewer is instructed to remove unsupported statements
instead of filling evidence gaps with guesses. The focused and full project
test suites pass; CrewAI also constructs all four agents and their four tasks
successfully.

The live end-to-end run with the local Ollama model did not finish: CrewAI
returned a `TimeoutError` before the final study pack was produced. The
checked-in pack is therefore a prepared example, not output from that live
run. Public search can also fail or return weak sources, and an LLM may
overlook a contradiction or produce an incomplete answer key. Tool errors are
reported in tool results rather than hidden, but successful search alone does
not prove a claim is true.

Next, I would profile and shorten local-model prompts, then rerun the
end-to-end workflow and confirm fresh output timestamps. I would evaluate
claim-level citation support and quiz-answer accuracy against instructor-
reviewed examples, add human review for uncertain claims, and compare
sequential with hierarchical orchestration. Students should independently
verify cited sources before using generated material for an exam.
