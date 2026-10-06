"""Offline pipeline and Streamlit interaction tests for the final UI."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.genai import types
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from streamlit.testing.v1 import AppTest

import app
from agent.agent import StudyAgent
from agent.llm import ask_llm, get_api_key, use_api_key
from agent.retriever import NotesIndex


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block live Gemini calls even when the developer has a local .env key."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("agent.agent.genai.Client", MagicMock(
        side_effect=RuntimeError("Offline test: no live Gemini calls")
    ))
    monkeypatch.setattr("agent.qa.ask_llm", MagicMock(return_value="ATP stores energy. (Page 1)"))
    monkeypatch.setattr("agent.generators.ask_llm", MagicMock(return_value="[]"))


def sample_pdf(text: str = "ATP stores energy. Cells use ATP for respiration.") -> bytes:
    """Build a real one-page PDF without optional sample files or reportlab."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
    })
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 700 Td ({safe}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = stream
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def reply(task: str, output: str | list, tool: str) -> dict:
    """Supply UI fixtures using the exact backend result shape."""
    return {"task": task, "output": output, "tool_used": tool,
            "sources": [{"id": 0, "page": 1, "text": "ATP stores energy."}],
            "steps": [{"thought_or_tool": tool, "input": {}, "observation": output}]}


def quiz() -> list[dict]:
    """Use four distinct options to exercise selection and score display."""
    return [{"question": "What stores energy?", "options": ["ATP", "Water", "Salt", "Air"],
             "answer_index": 0, "explanation": "ATP is the energy currency."}]


def test_import_and_helpers() -> None:
    """Importing the app exposes helpers without starting UI or requiring keys."""
    assert callable(app.main) and callable(app.run_request)
    assert app.score_quiz(quiz(), [0]) == 1
    assert app.score_quiz(quiz(), [None]) == 0
    assert app.tools_used(reply("quiz", quiz(), "make_quiz")) == {"make_quiz"}


@pytest.mark.parametrize("output", ["Error: API", [{"error": "Error: API"}], ["Error: API"]])
def test_error_sentinels(output: str | list) -> None:
    """Recognize all HANDOFF failure shapes before indexing structured fields."""
    assert app.output_error(output) == "Error: API"
    assert app.output_error(quiz()) is None


def test_full_pdf_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Read a real PDF, build the real index and run fallback Q&A with ask_llm mocked."""
    path = tmp_path / "notes.pdf"
    path.write_bytes(sample_pdf())
    pages, chunks, index = app.build_notes(path.read_bytes())
    llm = MagicMock(return_value="ATP stores energy. (Page 1)")
    monkeypatch.setattr("agent.qa.ask_llm", llm)
    result = StudyAgent(index).run("What is ATP?")
    assert pages[0]["page"] == chunks[0]["page"] == result["sources"][0]["page"] == 1
    assert result["task"] == "qa" and result["output"] == llm.return_value
    assert "ATP stores energy" in llm.call_args.kwargs["prompt"]
    assert "search_notes" in app.tools_used(result)


def test_function_calling_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise real agent routing plus quiz generation with both model layers mocked."""
    _, _, index = app.build_notes(sample_pdf())
    client = MagicMock()
    client.__enter__.return_value = client
    parts = [types.Part.from_function_call(name="make_quiz", args={"n": 1}),
             types.Part.from_text(text="Here is your quiz.")]
    client.models.generate_content.side_effect = [
        types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[part]))]) for part in parts
    ]
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr("agent.agent.genai.Client", constructor)
    llm = MagicMock(return_value=json.dumps(quiz()))
    monkeypatch.setattr("agent.generators.ask_llm", llm)
    result = app.run_request(index, "Make one quiz question", "session-key")
    constructor.assert_called_once_with(api_key="session-key")
    llm.assert_called_once()
    assert result["task"] == "quiz" and result["output"] == quiz()
    assert result["steps"][-1]["thought_or_tool"] == "final_response"
    assert get_api_key() == ""


def test_key_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keys reset after errors and never leak into another thread or environment."""
    with pytest.raises(RuntimeError), use_api_key("session-key"):
        assert get_api_key() == "session-key"
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(get_api_key).result() == ""
        with use_api_key("nested-key"):
            assert get_api_key() == "nested-key"
        assert get_api_key() == "session-key"
        raise RuntimeError("simulate failed request")
    assert get_api_key() == ""
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    with use_api_key("sidebar-key"):
        assert get_api_key() == "environment-key"


def test_text_generation_uses_session_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unchanged ask_llm API must see the same key as the routing client."""
    client = MagicMock()
    client.models.generate_content.return_value.text = "A real-shaped response"
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr("agent.llm.genai.Client", constructor)
    with use_api_key("sidebar-key"):
        assert ask_llm("Explain ATP") == "A real-shaped response"
    constructor.assert_called_once_with(api_key="sidebar-key")


@pytest.mark.parametrize("text", ["", "   ", "!!!"])
def test_unreadable_pdf(text: str) -> None:
    """Blank/scanned-like and wordless PDFs give a useful validation error."""
    with pytest.raises(ValueError, match="readable text|indexable words"):
        app.build_notes(sample_pdf(text))


def test_retriever_regressions() -> None:
    """Repeated notes, external edits and nonpositive top_k must be safe."""
    chunks = [{"id": i, "page": i + 1, "text": "ATP stores energy"} for i in range(20)]
    index = NotesIndex(chunks)
    chunks[0]["text"] = "Unrelated replacement"
    assert all(item["text"] == "ATP stores energy" for item in index.search("ATP", 20))
    assert index.search("ATP", 0) == index.search("ATP", -1) == []


def ui(monkeypatch: pytest.MonkeyPatch, uploaded: io.BytesIO | None = None) -> AppTest:
    """Drive the actual app, substituting only the uploader's file transport."""
    # AppTest has no file-upload setter. A BytesIO has the same getvalue method
    # used by our UI; the real PDF parser, buttons and session state still run.
    monkeypatch.setattr(app.st, "file_uploader", MagicMock(return_value=uploaded))
    return AppTest.from_string("from app import main\nmain()", default_timeout=15).run()


def click(at: AppTest, label: str) -> AppTest:
    """Click a visible button by its user-facing label."""
    next(button for button in at.button if button.label == label).click()
    return at.run()


def ready_ui(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Process real test notes and supply a dummy environment key."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    at = ui(monkeypatch, io.BytesIO(sample_pdf()))
    click(at, "Process notes")
    assert not at.exception
    return at


def test_missing_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show friendly warnings for absent PDFs and absent credentials."""
    at = ui(monkeypatch)
    click(at, "Process notes")
    assert "Upload a PDF first" in at.warning[0].value
    click(at, "Quiz")
    assert any("Upload a PDF" in item.value for item in at.warning)
    at = ui(monkeypatch, io.BytesIO(sample_pdf()))
    click(at, "Process notes")
    click(at, "Summary")
    assert any("Gemini API key" in item.value for item in at.warning)
    assert not at.exception


def test_quiz_interaction_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quiz grading and reruns preserve history and never regenerate the reply."""
    runner = MagicMock(return_value=reply("quiz", quiz(), "make_quiz"))
    monkeypatch.setattr(app, "run_request", runner)
    at = ready_ui(monkeypatch)
    index = at.session_state["index"]
    click(at, "Quiz")
    click(at, "Check answers")
    assert any("Choose an answer" in item.value for item in at.warning)
    at.radio[0].set_value("B. Water")
    click(at, "Check answers")
    assert any("Score: 0 / 1" in item.value for item in at.success)
    at.radio[0].set_value("A. ATP")
    click(at, "Check answers")
    assert any("Score: 1 / 1" in item.value for item in at.success)
    at.run()
    assert runner.call_count == 1 and len(at.session_state["messages"]) == 2
    assert at.session_state["index"] is index
    assert at.session_state["used_tools"] == {"make_quiz"}
    assert any("Generate Quiz — ✓ Used" in item.value for item in at.markdown)
    assert not at.exception


@pytest.mark.parametrize(("label", "task", "output", "tool"), [
    ("Summary", "summary", "A concise summary.", "summarize"),
    ("Important topics", "topics", ["ATP", "Respiration"], "important_topics"),
    ("Flashcards", "flashcards", [{"front": "ATP?", "back": "Energy currency"}], "make_flashcards"),
])
def test_output_renderers(monkeypatch: pytest.MonkeyPatch, label: str, task: str,
                          output: str | list, tool: str) -> None:
    """Render each quick action, including generator sources without scores."""
    runner = MagicMock(return_value=reply(task, output, tool))
    monkeypatch.setattr(app, "run_request", runner)
    at = ready_ui(monkeypatch)
    click(at, label)
    assert not at.exception
    assert any(f"Agent chose: {task}" in item.value for item in at.caption)
    assert {"Sources (pages)", "Agent steps"}.issubset({item.label for item in at.expander})
    if task == "summary":
        assert at.get("download_button")[0].label == "Download summary (.txt)"
    elif task == "topics":
        at.checkbox[0].check().run()
        assert at.checkbox[0].value
    else:
        assert "ATP?" in at.expander[0].label
        assert any("Energy currency" in item.value for item in at.markdown)
    at.run()
    runner.assert_called_once()


@pytest.mark.parametrize(("task", "output", "tool"), [
    ("quiz", [{"error": "Error: quota exceeded"}], "make_quiz"),
    ("flashcards", [{"error": "Error: invalid JSON"}], "make_flashcards"),
    ("topics", ["Error: quota exceeded"], "important_topics"),
    ("summary", "Error: service unavailable", "summarize"),
])
def test_failed_outputs(monkeypatch: pytest.MonkeyPatch, task: str,
                        output: str | list, tool: str) -> None:
    """Failure sentinels show errors and steps instead of crashing widgets."""
    monkeypatch.setattr(app, "run_request", MagicMock(return_value=reply(task, output, tool)))
    at = ready_ui(monkeypatch)
    click(at, "Quiz")
    assert at.error and not at.exception and not at.radio
    assert any(item.label == "Agent steps" for item in at.expander)


def test_chat_steps_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat input retains source citations, actual multi-tool steps and failures."""
    result = reply("qa", "ATP stores energy. The answer is 8.", "search_notes")
    result["sources"][0]["score"] = 0.9
    result["steps"] += [
        {"thought_or_tool": "calculator", "input": {"expression": "2*4"}, "observation": "8"},
        {"thought_or_tool": "final_response", "input": "ATP?", "observation": result["output"]},
    ]
    runner = MagicMock(side_effect=[result, RuntimeError("transport failed")])
    monkeypatch.setattr(app, "run_request", runner)
    at = ready_ui(monkeypatch)
    at.chat_input[0].set_value("Find ATP and calculate 2*4").run()
    assert any("Step 1: search_notes → Step 2: calculator → Step 3: Final answer" in item.value
               for item in at.caption)
    assert any("Page 1 · relevance 0.900" in item.value for item in at.markdown)
    at.chat_input[0].set_value("Another question").run()
    assert at.error and not at.exception
    assert len(at.session_state["messages"]) == 4


def test_upload_change_and_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing/removing a file invalidates its index, chat and quiz widgets."""
    monkeypatch.setattr(app, "run_request", MagicMock(return_value=reply("quiz", quiz(), "make_quiz")))
    at = ready_ui(monkeypatch)
    click(at, "Quiz")
    at.radio[0].set_value("A. ATP")
    click(at, "Check answers")
    monkeypatch.setattr(app.st, "file_uploader", MagicMock(return_value=io.BytesIO(sample_pdf("DNA stores genes."))))
    at.run()
    assert at.session_state["index"] is None and not at.session_state["messages"]
    assert not at.radio
    click(at, "Process notes")
    assert at.session_state["index"].search("DNA")
    monkeypatch.setattr(app.st, "file_uploader", MagicMock(return_value=None))
    at.run()
    assert at.session_state["index"] is None and not at.exception


def test_sidebar_key_and_invalid_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """The password fallback reaches requests; corrupt/blank PDFs remain unready."""
    runner = MagicMock(return_value=reply("summary", "Notes summary", "summarize"))
    monkeypatch.setattr(app, "run_request", runner)
    at = ui(monkeypatch, io.BytesIO(sample_pdf()))
    click(at, "Process notes")
    at.text_input[0].set_value("sidebar-key").run()
    click(at, "Summary")
    assert runner.call_args.args[2] == "sidebar-key"
    assert get_api_key() == ""
    for payload in (b"not a PDF", sample_pdf("")):
        monkeypatch.setattr(app.st, "file_uploader", MagicMock(return_value=io.BytesIO(payload)))
        click(at, "Process notes")
        assert at.error and at.session_state["index"] is None and not at.exception
