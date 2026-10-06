"""Completed Part 4 UI; all study requests go through StudyAgent.run."""

import hashlib
import io
import os
from collections.abc import MutableMapping
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent.agent import StudyAgent
from agent.chunker import chunk_pages
from agent.llm import use_api_key
from agent.pdf_loader import load_pdf
from agent.retriever import NotesIndex

load_dotenv()

QUICK_ACTIONS = {
    "Summary": "Summarize my notes",
    "Quiz": "Make 5 quiz questions from my notes",
    "Important topics": "List the important topics in my notes",
    "Flashcards": "Create 8 flashcards from my notes",
}
TOOLBOX = {
    "search_notes": "Search Notes",
    "calculator": "Calculator",
    "web_search": "Web Search",
    "make_quiz": "Generate Quiz",
    "summarize": "Summary",
    "important_topics": "Important Topics",
    "make_flashcards": "Flashcards",
}
API_ERROR = (
    "The request could not be completed. Check your Gemini key, quota, model "
    "access and internet connection, then try again."
)


def build_notes(pdf_bytes: bytes) -> tuple[list[dict], list[dict], NotesIndex]:
    """Run the existing PDF pipeline once and reject unreadable documents."""
    # A fresh byte stream avoids depending on the uploader's current cursor.
    pages = load_pdf(io.BytesIO(pdf_bytes))
    if not pages:
        raise ValueError(
            "No readable text found. This PDF may be scanned or blank. "
            "Try a text-based PDF or apply OCR first."
        )
    chunks = chunk_pages(pages)
    try:
        index = NotesIndex(chunks)
    except ValueError as exc:
        raise ValueError("The PDF has no indexable words. Try different notes.") from exc
    return pages, chunks, index


def initialize_state(state: MutableMapping[str, Any]) -> None:
    """Create per-browser defaults without erasing values during reruns."""
    # Streamlit runs this script again after every button click. setdefault
    # preserves the expensive index, previous replies and quiz selections.
    for key, value in {
        "upload_digest": None, "index": None, "pages": [], "chunks": [],
        "messages": [], "used_tools": set(),
    }.items():
        state.setdefault(key, value)


def sync_upload(state: MutableMapping[str, Any], digest: str | None) -> None:
    """Invalidate document-specific state when a PDF is changed or removed."""
    if state["upload_digest"] == digest:
        return
    state.update(upload_digest=digest, index=None, pages=[], chunks=[], messages=[])
    # Widget keys outlive chat messages. Remove them so a new PDF cannot inherit
    # the previous document's quiz answers or topic checkmarks.
    for key in list(state):
        if str(key).startswith("reply_"):
            del state[key]
    # used_tools intentionally survives: the panel describes the whole session.


def output_error(output: Any) -> str | None:
    """Recognize every failure payload documented in HANDOFF.md."""
    if isinstance(output, str) and output.startswith("Error:"):
        return output
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and "error" in item:
                return str(item["error"])
            if isinstance(item, str) and item.startswith("Error:"):
                return item
    return None


def tools_used(result: dict) -> set[str]:
    """Read actual tool attempts from the trace, including fallback Q&A."""
    actions = {step["thought_or_tool"] for step in result.get("steps", [])}
    if "answer_question" in actions:
        actions.add("search_notes")  # This backend function performs retrieval.
    return actions.intersection(TOOLBOX)


def score_quiz(questions: list[dict], answers: list[int | None]) -> int:
    """Count correct choices; an unanswered question never earns a point."""
    return sum(answer == question["answer_index"]
               for question, answer in zip(questions, answers))


def run_request(index: NotesIndex, message: str, api_key: str = "") -> dict:
    """Call the existing agent with a request-scoped sidebar key."""
    # Do not set os.environ here: it is shared by every browser session.
    with use_api_key(api_key):
        return StudyAgent(index).run(message)


def render_quiz(questions: list[dict], prefix: str) -> None:
    """Keep each quiz interactive and remember its submitted score on reruns."""
    # A form batches all choices into one submission. Stable, reply-specific
    # keys prevent two quizzes with identical questions from sharing answers.
    with st.form(f"{prefix}_form"):
        answers = []
        for number, question in enumerate(questions):
            choices = [f"{letter}. {text}" for letter, text
                       in zip("ABCD", question["options"])]
            choice = st.radio(
                f"{number + 1}. {question['question']}", choices, index=None,
                key=f"{prefix}_choice_{number}",
            )
            answers.append(choices.index(choice) if choice is not None else None)
        submitted = st.form_submit_button("Check answers")
    if submitted:
        if any(answer is None for answer in answers):
            st.warning("Choose an answer for every question before checking.")
        else:
            # Store a snapshot so changing an unsubmitted choice cannot silently
            # change the score that belongs to the last checked set of answers.
            st.session_state[f"{prefix}_checked"] = answers
    checked = st.session_state.get(f"{prefix}_checked")
    if checked is not None:
        st.success(f"Score: {score_quiz(questions, checked)} / {len(questions)}")
        st.caption("Results for your last checked answers. Submit again to update.")
        for number, (question, answer) in enumerate(zip(questions, checked), 1):
            correct = question["answer_index"]
            marker = "✓" if answer == correct else "✗"
            st.write(f"{marker} {number}. Correct answer: {question['options'][correct]}")
            st.caption(question["explanation"])


def render_steps(result: dict) -> None:
    """Show recorded actions and observations, without inventing agent steps."""
    with st.expander("Agent steps"):
        steps = result.get("steps", [])
        labels = []
        for number, step in enumerate(steps, 1):
            action = step["thought_or_tool"]
            label = "Final answer" if action == "final_response" else action
            labels.append(f"Step {number}: {label}")
        # The fallback backend returns directly without a final_response event.
        if not steps or steps[-1]["thought_or_tool"] != "final_response":
            labels.append("Final answer" if not output_error(result["output"]) else "Request failed")
        st.caption(" → ".join(labels))
        st.caption("Recorded tool actions and results; not private model reasoning.")
        for label, step in zip(labels, steps):
            st.markdown(f"**{label}**")
            st.json({"input": step["input"], "observation": step["observation"]})


def render_reply(result: dict, reply_id: int) -> None:
    """Render the agent's documented task, output, sources and visible trace."""
    task, output = result["task"], result["output"]
    prefix = f"reply_{reply_id}"
    st.caption(f"🤖 Agent chose: {task} | tool: {result['tool_used']}")
    error = output_error(output)
    if error:
        # Error sentinels are not quiz questions, card faces or topic names.
        st.error(API_ERROR)
        st.text(error)
    elif task == "quiz":
        render_quiz(output, prefix)
    elif task == "flashcards":
        for number, card in enumerate(output, 1):
            with st.expander(f"{number}. {card['front']} — reveal answer"):
                st.write(card["back"])
    elif task == "topics":
        st.caption("Tick each topic as you revise it.")
        for number, topic in enumerate(output):
            st.checkbox(topic, key=f"{prefix}_topic_{number}")
    else:
        st.markdown(output)
        if task == "summary":
            st.download_button(
                "Download summary (.txt)", data=output,
                file_name="study_summary.txt", mime="text/plain",
                key=f"{prefix}_download",
            )
    if task == "qa" or result.get("sources"):
        with st.expander("Sources (pages)"):
            sources = result.get("sources", [])
            if not sources:
                st.caption("No source passages were returned for this reply.")
            for source in sources:
                score = source.get("score")  # Generator sources have no score.
                suffix = f" · relevance {score:.3f}" if score is not None else ""
                st.markdown(f"**Page {source['page']}{suffix}**")
                st.text(source["text"])
    render_steps(result)


def main() -> None:
    """Build the sidebar and chat while caching all document data per session."""
    st.set_page_config(page_title="Study Agent", page_icon="🎓", layout="wide")
    initialize_state(st.session_state)
    st.title("🎓 Study Agent")
    st.write("Turn your PDF notes into answers, quizzes and revision material.")
    quick_prompt = None
    with st.sidebar:
        st.header("Your notes")
        uploaded = st.file_uploader("Upload PDF notes", type=["pdf"])
        pdf_bytes = uploaded.getvalue() if uploaded is not None else None
        digest = hashlib.sha256(pdf_bytes).hexdigest() if pdf_bytes is not None else None
        sync_upload(st.session_state, digest)
        if st.button("Process notes", type="primary"):
            if pdf_bytes is None:
                st.warning("Upload a PDF first, then click Process notes.")
            elif st.session_state.index is not None:
                st.info("These notes are already processed and ready.")
            else:
                try:
                    with st.spinner("Reading and indexing your notes…"):
                        pages, chunks, index = build_notes(pdf_bytes)
                    # Save only a complete index; a failed upload stays unready.
                    st.session_state.update(pages=pages, chunks=chunks, index=index)
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("Could not read this PDF. Try an unencrypted, valid PDF.")
        if st.session_state.index is not None:
            st.success(f"Ready: {len(st.session_state.pages)} text pages · "
                       f"{len(st.session_state.chunks)} chunks")
        env_key = os.environ.get("GEMINI_API_KEY", "").strip()
        fallback_key = ""
        if env_key:
            st.caption("Gemini API key loaded from environment / .env.")
        else:
            fallback_key = st.text_input(
                "Gemini API key", type="password", key="session_api_key",
                help="Used only for this session; not saved to .env.",
            ).strip()
        st.caption("Requests send selected note excerpts to Gemini.")
        st.subheader("Quick actions")
        for label, prompt in QUICK_ACTIONS.items():
            if st.button(label, use_container_width=True):
                quick_prompt = prompt
        # Populate this after handling the request so usage updates immediately.
        tools_panel = st.empty()

    if st.session_state.index is None:
        st.info("Upload a text-based PDF and click Process notes to begin.")
    for number, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_reply(message["result"], number)
    typed_prompt = st.chat_input("Ask about your notes, or request a study activity…")
    prompt = quick_prompt or typed_prompt
    if prompt:
        if st.session_state.index is None:
            st.warning("Upload a PDF and click Process notes before asking.")
        elif not (env_key or fallback_key):
            st.warning("Add your Gemini API key in the sidebar or in .env first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Study Agent is working…"):
                        result = run_request(st.session_state.index, prompt, fallback_key)
                except Exception:
                    # Unexpected transport failures still leave a readable reply
                    # in history, without exposing exception internals or secrets.
                    result = {
                        "task": "qa", "output": f"Error: {API_ERROR}",
                        "tool_used": "unavailable", "sources": [], "steps": [],
                    }
                st.session_state.used_tools.update(tools_used(result))
                reply_id = len(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "result": result})
                render_reply(result, reply_id)
    with tools_panel.container():
        st.subheader("Agent tools")
        st.caption("Used = attempted during this session, including failed tools.")
        for name, label in TOOLBOX.items():
            status = "✓ Used" if name in st.session_state.used_tools else "○ Not used"
            st.write(f"{label} — {status}")
        st.caption("Web Search is optional: install ddgs to enable it.")


# Importing helpers in tests must not start Streamlit or call Gemini.
if __name__ == "__main__":
    main()
