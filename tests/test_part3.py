"""Offline Part 3 tests: mock both text generation and Gemini tool responses."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import types

from agent.agent import StudyAgent, classify_task
from agent.generators import (
    important_topics, make_flashcards, make_quiz, parse_json,
    representative_chunks, summarize,
)
from agent.retriever import NotesIndex
from agent.tools import calculator, search_notes, web_search


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure every test uses fake models, never credentials or network access."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("agent.generators.ask_llm", MagicMock(return_value="[]"))
    monkeypatch.setattr("agent.qa.ask_llm", MagicMock(return_value="Mock Q&A"))
    monkeypatch.setattr("agent.agent.genai.Client", MagicMock(
        side_effect=RuntimeError("No live API calls allowed in tests")
    ))


@pytest.fixture
def index() -> NotesIndex:
    """Provide real searchable chunks spread across different pages."""
    return NotesIndex([
        {"id": 0, "page": 1, "text": "Photosynthesis uses light and chlorophyll."},
        {"id": 1, "page": 2, "text": "Respiration produces ATP energy."},
        {"id": 2, "page": 3, "text": "DNA replication copies genetic information."},
    ])


@pytest.mark.parametrize(("expression", "expected"), [
    ("2 + 3 * 4", "14"), ("(2 + 3) * 4", "20"), ("2 ** 3", "8"),
    ("10 % 3", "1"), ("9 / 2", "4.5"), ("-5 + +2", "-3"),
    ("2 ** -2", "0.25"),
])
def test_calculator(expression: str, expected: str) -> None:
    """Check precedence, parentheses and every supported operator."""
    assert calculator(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os')", "open('secret')", "(1).__class__", "x + 1",
    "[1, 2]", "True + 1", "'2' * 3", "1 // 2", "1 << 2",
    "1 / 0", "9 ** 999999999", "(-1) ** 0.5", "1e309", "",
    "2+" , "9" * 301, "+".join(["1"] * 60),
])
def test_calculator_rejects_unsafe_or_invalid(expression: str) -> None:
    """Reject Python execution, invalid arithmetic and excessive resources."""
    assert calculator(expression).startswith("Error:")


@pytest.mark.parametrize(("message", "task"), [
    ("Summarize my notes", "summary"), ("Give an overview", "summary"),
    ("Make 5 quiz questions", "quiz"), ("Test me on the notes", "quiz"),
    ("Make flashcards", "flashcards"), ("Create 3 flash cards", "flashcards"),
    ("What are the important topics?", "topics"), ("List key topics", "topics"),
    ("Calculate (2 + 3) * 4", "calculate"), ("2 ** 4", "calculate"),
    ("Search the web for biology", "web"), ("Find the latest news online", "web"),
    ("Explain photosynthesis", "qa"), ("What is ATP?", "qa"),
])
def test_router(message: str, task: str) -> None:
    """Verify fallback classification with fourteen student messages."""
    assert classify_task(message) == task


def test_json_fences_and_schema() -> None:
    """Allow surrounding JSON fences and enforce each required key."""
    card = [{"front": "ATP?", "back": "Energy currency"}]
    assert parse_json("```json\n" + json.dumps(card) + "\n```", "flashcards", 1) == card
    assert parse_json('["DNA", "ATP"]', "topics") == ["DNA", "ATP"]
    quiz = [{"question": "ATP?", "options": ["a", "b", "c", "d"],
             "answer_index": 1, "explanation": "Energy"}]
    assert parse_json(json.dumps(quiz), "quiz", 1) == quiz


@pytest.mark.parametrize(("payload", "kind", "n"), [
    ("not json", "topics", None), ('{"topics": []}', "topics", None),
    ('[""]', "topics", None), ('[2]', "topics", None),
    ('[{"front": "Q"}]', "flashcards", 1),
    ('[{"front": "Q", "back": "A", "extra": 2}]', "flashcards", 1),
    ('[{"front": "Q", "back": "A"}]', "flashcards", 2),
    ('[{"question":"Q","options":["a","b","c"],"answer_index":0,"explanation":"E"}]',
     "quiz", 1),
    ('[{"question":"Q","options":["a","b","c","d"],"answer_index":true,"explanation":"E"}]',
     "quiz", 1),
    ('[{"question":"Q","options":["a","b","c","d"],"answer_index":4,"explanation":"E"}]',
     "quiz", 1),
])
def test_invalid_json(payload: str, kind: str, n: int | None) -> None:
    """Reject parseable JSON whose field types or item count are incorrect."""
    with pytest.raises((ValueError, TypeError)):
        parse_json(payload, kind, n)


def test_retry_and_error_shapes(index: NotesIndex, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry once, then keep the expected outer return type on failure."""
    llm = MagicMock(side_effect=["bad", '[{"front":"Q","back":"A"}]'])
    monkeypatch.setattr("agent.generators.ask_llm", llm)
    assert make_flashcards(index, 1) == [{"front": "Q", "back": "A"}]
    assert llm.call_count == 2
    llm.side_effect = None
    llm.return_value = "bad"
    assert "error" in make_quiz(index)[0]
    assert important_topics(index)[0].startswith("Error:")
    llm.side_effect = RuntimeError("Service unavailable")
    assert "error" in make_flashcards(index)[0]
    assert summarize(index).startswith("Error:")


def test_generators_success(index: NotesIndex, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure production generators return the actual model data."""
    monkeypatch.setattr("agent.generators.ask_llm", MagicMock(return_value='["ATP"]'))
    assert important_topics(index) == ["ATP"]
    monkeypatch.setattr("agent.generators.ask_llm", MagicMock(return_value="Short summary"))
    assert summarize(index) == "Short summary"
    quiz = [{"question": "ATP?", "options": ["a", "b", "c", "d"],
             "answer_index": 0, "explanation": "Energy"}]
    monkeypatch.setattr("agent.generators.ask_llm", MagicMock(return_value=json.dumps(quiz)))
    assert make_quiz(index, 1) == quiz


def test_whole_document_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Include the ending of long notes without retrieving just one topic."""
    chunks = [{"id": i, "page": i + 1, "text": f"Unique topic {i}"} for i in range(60)]
    index = NotesIndex(chunks)
    selected = representative_chunks(index)
    assert len(selected) == 12
    assert selected[0]["id"] == 0 and selected[-1]["id"] == 59
    assert len({chunk["id"] for chunk in selected}) == 12
    llm = MagicMock(return_value="Summary")
    monkeypatch.setattr("agent.generators.ask_llm", llm)
    summarize(index)
    assert "Unique topic 59" in llm.call_args.args[0]
    selected[0]["text"] = "Changed copy"
    assert chunks[0]["text"] == "Unique topic 0"


def test_empty_notes_and_bad_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip Gemini calls when there is no context or the count is invalid."""
    llm = MagicMock()
    monkeypatch.setattr("agent.generators.ask_llm", llm)
    index = NotesIndex([])
    assert summarize(index).startswith("Error:")
    assert "error" in make_quiz(index)[0]
    assert "error" in make_flashcards(index, -1)[0]
    assert "error" in make_quiz(index, True)[0]
    assert important_topics(index)[0].startswith("Error:")
    llm.assert_not_called()


def test_search_wrapper(index: NotesIndex) -> None:
    """Preserve raw source dicts and relevance scores."""
    assert search_notes("ATP", index) == index.search("ATP")


def test_optional_web(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing DDGS, success and failure without real web access."""
    import sys
    monkeypatch.setitem(sys.modules, "ddgs", None)
    assert web_search("biology") == (
        "Web search is not enabled. Please enable it by configuring a search API key."
    )
    ddgs = MagicMock()
    ddgs.return_value.text.return_value = [{"title": "Biology", "body": "Notes", "href": "https://example.com"}]
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=ddgs))
    assert "https://example.com" in web_search("biology")
    ddgs.return_value.text.side_effect = RuntimeError("Offline")
    assert web_search("biology").startswith("Error:")


def _response(*parts: types.Part) -> types.GenerateContentResponse:
    """Build real SDK response objects so tests verify message serialization."""
    return types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(role="model", parts=list(parts)))
    ])


def _client(monkeypatch: pytest.MonkeyPatch, responses: list) -> MagicMock:
    """Install a fake context-managed SDK client and a dummy key."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.side_effect = responses
    monkeypatch.setattr("agent.agent.genai.Client", MagicMock(return_value=client))
    return client


def test_multistep_agent(index: NotesIndex, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Search, calculate and respond in separate actual loop iterations."""
    client = _client(monkeypatch, [
        _response(types.Part.from_function_call(name="search_notes", args={"query": "ATP"})),
        _response(types.Part.from_function_call(name="calculator", args={"expression": "2 * 4"})),
        _response(types.Part.from_text(text="The answer is 8.")),
    ])
    result = StudyAgent(index).run("Find ATP and calculate 2 * 4")
    assert result["task"] == "calculate" and result["output"] == "The answer is 8."
    assert result["sources"][0]["page"] == 2
    assert [step["thought_or_tool"] for step in result["steps"]] == [
        "search_notes", "calculator", "final_response"
    ]
    assert "[step 3]" in capsys.readouterr().out
    assert client.models.generate_content.call_count == 3
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert contents[2].parts[0].function_response.response["result"]
    names = {d.name for d in client.models.generate_content.call_args.kwargs["config"].tools[0].function_declarations}
    assert names == {"search_notes", "calculator", "web_search", "make_quiz",
                     "make_flashcards", "summarize", "important_topics"}


def test_parallel_calls_and_bad_arguments(index: NotesIndex, monkeypatch: pytest.MonkeyPatch) -> None:
    """Send all tool responses together and let Gemini recover from bad calls."""
    _client(monkeypatch, [
        _response(
            types.Part(function_call=types.FunctionCall(
                name="calculator", args={"expression": "3 + 4"}, id="call-1")),
            types.Part.from_function_call(name="calculator", args={"expression": 7}),
            types.Part.from_function_call(name="unknown", args={}),
        ),
        _response(types.Part.from_text(text="7")),
    ])
    result = StudyAgent(index).run("Calculate 3 + 4")
    assert result["output"] == "7"
    assert len(result["steps"]) == 4


def test_structured_output_preserved(index: NotesIndex, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep quiz arrays for the UI even when Gemini's final turn is prose."""
    cards = [{"front": "ATP?", "back": "Energy currency"}]
    monkeypatch.setattr("agent.generators.ask_llm", MagicMock(return_value=json.dumps(cards)))
    _client(monkeypatch, [
        _response(types.Part.from_function_call(name="make_flashcards", args={"n": 1})),
        _response(types.Part.from_text(text="Here is your card.")),
    ])
    result = StudyAgent(index).run("Make one flashcard")
    assert result["output"] == cards and result["task"] == "flashcards"
    assert set(result) == {"task", "output", "sources", "tool_used", "steps"}


def test_fallback_and_qa_reuse(index: NotesIndex) -> None:
    """Without a key, use the existing Q&A function and safe calculator."""
    result = StudyAgent(index).run("What is ATP?")
    assert result["output"] == "Mock Q&A" and result["sources"]
    assert result["steps"][0]["thought_or_tool"] == "fallback_router"
    assert StudyAgent(index).run("Calculate (2 + 3) * 4")["output"] == "20"
    assert StudyAgent(index).run("")["output"].startswith("Error:")


def test_loop_limit(index: NotesIndex, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop after five model turns and record why the fallback was needed."""
    client = _client(monkeypatch, [
        _response(types.Part.from_function_call(name="calculator", args={"expression": "1 + 1"}))
        for _ in range(5)
    ])
    result = StudyAgent(index).run("Calculate 1 + 1")
    assert client.models.generate_content.call_count == 5
    assert result["output"] == "2"
    assert any("five-iteration" in str(step["observation"]) for step in result["steps"])


def test_empty_response_falls_back(index: NotesIndex, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed Gemini responses also activate the router instead of crashing."""
    _client(monkeypatch, [types.GenerateContentResponse()])
    assert StudyAgent(index).run("Calculate 2 + 2")["output"] == "4"
