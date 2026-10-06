# Study Agent 🎓

An AI-powered study assistant where students upload PDF notes and interact with them via natural language. Ask questions, get summaries, generate quizzes, discover important topics, and create flashcards — all powered by Google Gemini.

**Progress: Parts 1, 2 and 3 are complete. Part 4 (Streamlit UI) is next.**

---

## Project Structure

```
study-agent/
├── app.py                 # Streamlit UI (Part 4)
├── requirements.txt       # All dependencies
├── .env.example           # Copy to .env and fill your key
├── .gitignore
├── README.md
├── CONTRACT.md            # Binding API contracts for all parts
├── HANDOFF.md             # Progress log across all 4 parts
├── agent/
│   ├── __init__.py
│   ├── pdf_loader.py      # PDF → pages  (Part 1 ✅)
│   ├── chunker.py         # Pages → chunks (Part 1 ✅)
│   ├── retriever.py       # TF-IDF search index (Part 2 ✅)
│   ├── llm.py             # Gemini wrapper (Part 2 ✅)
│   ├── qa.py              # Q&A pipeline (Part 2 ✅)
│   ├── tools.py           # Notes search, calculator, web search (Part 3 ✅)
│   ├── generators.py      # Summary, quiz, topics, flashcards (Part 3 ✅)
│   └── agent.py           # Gemini tool-calling agent (Part 3 ✅)
├── tests/
│   ├── test_part1.py      # pytest tests for Part 1 ✅
│   ├── test_part2.py      # pytest tests for Part 2 ✅
│   └── test_part3.py      # pytest tests for Part 3 ✅
├── scripts/
│   ├── try_qa.py          # Q&A CLI (Part 2 ✅)
│   └── try_agent.py       # Interactive agent CLI (Part 3 ✅)
└── samples/               # Place a sample PDF here for testing
```

---

## Features

| Feature | Status |
|---|---|
| PDF loading & parsing | ✅ Part 1 |
| Text chunking | ✅ Part 1 |
| TF-IDF search / retrieval | ✅ Part 2 |
| Gemini LLM integration | ✅ Part 2 |
| Q&A over notes | ✅ Part 2 |
| Summarisation | ✅ Part 3 |
| Quiz generation | ✅ Part 3 |
| Important topics extraction | ✅ Part 3 |
| Flashcard generation | ✅ Part 3 |
| Notes search, safe calculator & optional web search | ✅ Part 3 |
| Gemini function-calling loop & visible tool steps | ✅ Part 3 |
| Streamlit UI | ⏳ Part 4 — next |

---

## Quick Start

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd study-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up the API key

```bash
cp .env.example .env
# Open .env and replace "your_key_here" with your actual Gemini API key
# Get one at: https://aistudio.google.com/app/apikey
```

### 5. Run the completed Part 3 CLI

```bash
python scripts/try_agent.py samples/my_notes.pdf
```

Use your own PDF path. Try `summarize notes`, `make 5 quiz questions`,
`create 8 flashcards`, or `calculate (12 + 8) / 4`. Tool steps print in the terminal.
Gemini 3.5 Flash-Lite handles tool selection when a valid API key is configured.

Web search is optional: install it with `pip install ddgs`.

### 6. Run the app after Part 4 is implemented

```bash
streamlit run app.py
```

### 7. Run tests

```bash
pytest tests/ -v
```

---

## Completed Parts and Next Handoff

Parts 1–3 are complete and ready for the Part 4 UI implementation.

### ✅ Part 1 Is Done — Here's What Was Built

- `agent/pdf_loader.py` — extracts text from PDF pages, skips blank pages
- `agent/chunker.py` — splits pages into overlapping text chunks, preserving page numbers
- `tests/test_part1.py` — full pytest suite for both modules
- `CONTRACT.md` — binding API contract all future parts must follow exactly
- `HANDOFF.md` — progress log

### ✅ Part 2 Is Done — Here's What Was Built

- `agent/retriever.py` — `NotesIndex` class: TF-IDF index (sklearn) + cosine-similarity search
- `agent/llm.py` — `ask_llm()`: Gemini wrapper (now using Gemini 3.5 Flash-Lite), retry logic, `MissingApiKeyError`
- `agent/qa.py` — `answer_question()`: full RAG pipeline (retrieve → prompt → generate)
- `tests/test_part2.py` — retrieval and Q&A tests, all run without an API key (LLM mocked)
- `scripts/try_qa.py` — CLI end-to-end tester: `python scripts/try_qa.py notes.pdf "question"`

### ✅ Part 3 Is Done — Here's What Was Built

- `agent/generators.py` — document-wide summary, quiz, important topics and flashcards; validated JSON with one retry
- `agent/tools.py` — notes search, safe AST calculator and optional DDGS web search
- `agent/agent.py` — Gemini function-calling loop (up to five turns), printed tool steps and fallback routing
- `tests/test_part3.py` — 61 offline tests with text generation and Gemini responses mocked
- `scripts/try_agent.py` — interactive PDF chat in the terminal
- `HANDOFF.md` — exact output shapes, error payloads and source formats for the UI

### 🔜 Hand Off to Part 4

Part 4 implements `app.py`, the Streamlit UI that ties the completed backend together.

1. Read `CONTRACT.md` and the completed Part 3 section in `HANDOFF.md`.
2. Reuse PDF loading, chunking and `NotesIndex`; create `StudyAgent(index)` after upload.
3. Call `StudyAgent.run(message)` and render its `task`, `output`, `sources`, `tool_used` and `steps` fields.
4. Render quizzes, flashcards and topics using their documented list formats, checking error sentinels first.
5. Show source pages and expandable tool steps; relevance scores are optional on sampled sources.
6. Keep credentials in the ignored local `.env`; do not put them in UI source or GitHub.
7. Add Part 4 tests and update this README and `HANDOFF.md` when the UI is complete.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key (required for Parts 2–4) |

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — web UI
- **Google Gemini** (`google-genai` SDK) — LLM backbone
- **pypdf** — PDF text extraction
- **scikit-learn** — TF-IDF similarity for retrieval
- **python-dotenv** — env variable loading
- **pytest** — testing

---

## Notes for Developers

- Never hardcode API keys — always use `.env` + `python-dotenv`
- Every module must be importable without crashing if `GEMINI_API_KEY` is missing
- Use type hints everywhere
- Write docstrings and heavy inline comments (this is a beginner-friendly project)
- All function signatures in `CONTRACT.md` are binding — do not change them
