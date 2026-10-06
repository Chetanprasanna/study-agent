# Study Agent 🎓

An AI-powered study assistant where students upload PDF notes and interact with them via natural language. Ask questions, get summaries, generate quizzes, discover important topics, and create flashcards — all powered by Google Gemini.

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
│   ├── retriever.py       # Semantic search index (Part 2)
│   ├── llm.py             # Gemini wrapper (Part 2)
│   ├── qa.py              # Q&A pipeline (Part 2)
│   ├── tools.py           # Calculator, web search (Part 3)
│   ├── generators.py      # Summary, quiz, flashcards (Part 3)
│   └── agent.py           # Orchestrator agent (Part 3)
├── tests/
│   └── test_part1.py      # pytest tests for Part 1 ✅
└── samples/               # Place a sample PDF here for testing
```

---

## Features

| Feature | Status |
|---|---|
| PDF loading & parsing | ✅ Part 1 |
| Text chunking | ✅ Part 1 |
| Semantic search / retrieval | Part 2 |
| Gemini LLM integration | Part 2 |
| Q&A over notes | Part 2 |
| Summarisation | Part 3 |
| Quiz generation | Part 3 |
| Important topics extraction | Part 3 |
| Flashcard generation | Part 3 |
| Streamlit UI | Part 4 |

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

### 5. Run the app (once Part 4 is done)

```bash
streamlit run app.py
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## What You Should Do After Part 1 Is Complete

> This section is for **you** — the project owner — to read before handing off to Part 2.

### ✅ Part 1 Is Done — Here's What Was Built

- `agent/pdf_loader.py` — extracts text from PDF pages, skips blank pages
- `agent/chunker.py` — splits pages into overlapping text chunks, preserving page numbers
- `tests/test_part1.py` — full pytest suite for both modules
- `CONTRACT.md` — binding API contract all future parts must follow exactly
- `HANDOFF.md` — progress log (Part 2 must append their own section)

### 🔜 Hand Off to Part 2

**Tell Part 2's Claude:**
1. Read `CONTRACT.md` first — it defines the exact function signatures they must implement.
2. Read `HANDOFF.md` — it describes what Part 1 built and what Part 2 needs to know.
3. They must implement: `agent/retriever.py`, `agent/llm.py`, `agent/qa.py`.
4. They must add their own section to `HANDOFF.md` when done.
5. They must write `tests/test_part2.py`.

### 🔜 Hand Off to Part 3

Part 3 implements `agent/tools.py`, `agent/generators.py`, `agent/agent.py`.

### 🔜 Hand Off to Part 4

Part 4 implements `app.py` — the Streamlit UI that ties everything together.

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
