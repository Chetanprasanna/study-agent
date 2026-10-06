# Study Agent — Handoff Log

Each part appends a new section to this file when complete.  
Read **all previous sections** before starting your part.

---

## Part 1 — Foundation & PDF Pipeline ✅

**Completed by:** Part 1  
**Date:** 2026-10-06  
**Files owned:**
- `agent/pdf_loader.py`
- `agent/chunker.py`
- `tests/test_part1.py`
- `CONTRACT.md`, `README.md`, `HANDOFF.md`, `requirements.txt`, `.env.example`, `.gitignore`
- Stub files for all other modules (importable, one-line docstrings)

---

### What Was Built

#### `agent/pdf_loader.py` — `load_pdf(file)`
- Accepts a file path (string / `pathlib.Path`) or a file-like object (e.g. Streamlit's `UploadedFile`).
- Uses `pypdf.PdfReader` to extract text from each page.
- Returns `list[dict]` where each dict is `{"page": int, "text": str}`.
- Skips pages where stripped text is empty.
- Page numbers are **1-indexed**.

#### `agent/chunker.py` — `chunk_pages(pages, chunk_size=800, overlap=150)`
- Accepts the list from `load_pdf()`.
- Splits each page's text into overlapping chunks of ~`chunk_size` characters.
- Tries to split on paragraph boundaries (`\n\n`) first, then sentence boundaries (`. `), then falls back to hard character split.
- Each chunk carries the `page` number it came from.
- Returns `list[dict]`: `{"id": int, "page": int, "text": str}`, globally 0-indexed `id`.

#### `tests/test_part1.py`
- Tests for `load_pdf`: valid PDF, file-like object, empty pages skipped, 1-indexed pages.
- Tests for `chunk_pages`: output shape, overlap, page tracking, edge cases (short text, empty list).

---

### How to Run Tests

```bash
# From the repo root
pip install -r requirements.txt
pytest tests/test_part1.py -v
```

---

### What Part 2 Must Know

1. **Read `CONTRACT.md` first** — it defines every function signature you must implement.
2. You need to implement: `agent/retriever.py`, `agent/llm.py`, `agent/qa.py`.
3. `NotesIndex` takes the output of `chunk_pages()` directly — the chunk dicts have keys `id`, `page`, `text`.
4. Build the TF-IDF index in `NotesIndex.__init__` using `sklearn.feature_extraction.text.TfidfVectorizer` and `sklearn.metrics.pairwise.cosine_similarity`.
5. `ask_llm()` must read `GEMINI_API_KEY` from the environment. Use `load_dotenv()` at module top. Do **not** raise on import if the key is missing — only raise inside `ask_llm()` when actually called.
6. Use `gemini-2.0-flash` (or later) as the default model — it's fast and free-tier friendly.
7. `answer_question()` must return `{"answer": str, "sources": list[dict]}` — the sources are the raw dicts from `index.search()` (already include `"score"`).
8. Write `tests/test_part2.py` and append your section to this `HANDOFF.md`.
9. All modules must remain importable without `GEMINI_API_KEY` set.

---

## Part 2 — Retrieval & LLM Integration ✅

**Completed by:** Part 2  
**Date:** 2026-10-06  
**Files owned:**
- `agent/retriever.py`
- `agent/llm.py`
- `agent/qa.py`
- `tests/test_part2.py`
- `scripts/try_qa.py`

---

### What Was Built

#### `agent/retriever.py` — `class NotesIndex`
- `__init__(chunks)`: Builds a TF-IDF index using `sklearn.TfidfVectorizer` with:
  - `sublinear_tf=True` (log-dampened term frequency)
  - `ngram_range=(1, 2)` (unigrams + bigrams for better matching)
  - `max_df=0.95` (ignore words in 95%+ of chunks — too generic to be useful)
- `search(query, top_k=4)`: Vectorises the query, computes cosine similarity
  against all chunk vectors, returns top-k results sorted descending by score.
- Returns **copies** of chunk dicts (never mutates originals) with an added `"score"` field.
- Handles empty index, empty query, and unknown words gracefully (returns `[]`).

#### `agent/llm.py` — `ask_llm(prompt, system=None)`
- Uses `google-genai` SDK with model `"gemini-2.5-flash"`.
- Reads `GEMINI_API_KEY` from env via `load_dotenv()` + `os.environ`.
- Raises `MissingApiKeyError` (custom exception, not crash-on-import) when key absent.
- Retries up to 2 times with exponential backoff (1s → 2s) on transient errors.
- Raises `RuntimeError` after all retries exhausted.
- **Safe to import without a key** — exception only thrown when `ask_llm()` is called.

#### `agent/qa.py` — `answer_question(question, index)`
- Retrieves top-4 relevant chunks via `index.search()`.
- Builds a structured RAG prompt with: labelled context excerpts (with page numbers
  and scores), the student's question, and explicit instructions to cite pages and
  say "I couldn't find this in the notes." when unanswerable.
- Passes a `system` instruction to Gemini to enforce faithful, citation-aware answers.
- Short-circuits with `{"answer": "I couldn't find this in the notes.", "sources": []}` 
  when the index is empty or no matches are found.

#### `tests/test_part2.py`
- **34 tests** — all run without `GEMINI_API_KEY`.
- `NotesIndex`: construction, ranking, top_k, edge cases, no-mutation guarantee.
- `ask_llm`: missing key error, retry logic, success path — all mocked.
- `answer_question`: return shape, source content, prompt content — LLM mocked.

#### `scripts/try_qa.py`
- CLI tool: `python scripts/try_qa.py <pdf_path> [question]`
- Runs the full pipeline end-to-end: load PDF → chunk → index → answer.
- Interactive loop mode (no question arg) and single-question mode.
- Colour-coded terminal output with source snippets and scores.

---

### How to Run Tests

```bash
# From repo root, with venv activated
pytest tests/test_part2.py -v
# Also make sure Part 1 tests still pass:
pytest tests/ -v
```

### How to Run the End-to-End CLI

```bash
# Single question:
python scripts/try_qa.py samples/my_notes.pdf "What is the role of mitochondria?"

# Interactive mode:
python scripts/try_qa.py samples/my_notes.pdf
```

---

### What Part 3 Must Know

1. **Read `CONTRACT.md`** — your function signatures are already defined there.
2. You need to implement: `agent/tools.py`, `agent/generators.py`, `agent/agent.py`.
3. **Import paths**: always use `from agent.retriever import NotesIndex`, 
   `from agent.llm import ask_llm`, `from agent.qa import answer_question`.
4. **`ask_llm` is your LLM call**: use it in generators.py for summarize/quiz/flashcards.
   It handles retries, so you don't need to add retry logic in generators.py.
5. **`MissingApiKeyError`** lives in `agent.llm`. Catch it in `agent.py` if you want
   to return a user-friendly dict instead of propagating the exception.
6. **`calculator()` in tools.py** must use `ast`-based safe evaluation — absolutely
   no `eval()` or `exec()`. See CONTRACT.md for the exact requirement.
7. **`StudyAgent.run()` return format** (CONTRACT.md §8):
   ```python
   {"task": str, "output": str|list, "sources": list[dict], "tool_used": str}
   ```
   Route by classifying the `user_message` string (keyword matching or LLM routing).
8. For generators: retrieve ALL chunks via `index._chunks` or search with a broad query
   to get enough context for summary/quiz/flashcard generation.
9. Write `tests/test_part3.py` and append your section to `HANDOFF.md`.
10. All modules must remain importable without `GEMINI_API_KEY` set.

### Known Limitations / Notes for Part 3

- TF-IDF retrieval is keyword-based, not semantic. If the user phrases their question
  very differently from the notes, retrieval may miss relevant chunks. Part 3's agent
  could re-phrase the query before searching to mitigate this.
- `ask_llm` currently has no token-count guard. For very large PDFs, the prompt 
  (context + question) could exceed Gemini's context window. Part 3 generators should
  limit context to a reasonable number of chunks (e.g. top-8 for summarisation).
- `gemini-2.5-flash` is the model used. If it's deprecated by the time you build Part 3,
  update `_DEFAULT_MODEL` in `agent/llm.py`.

---

## Part 3 — Generators, Tools & Agent Orchestrator

> *Part 3: append your section here when done.*

---

## Part 4 — Streamlit UI

> *Part 4: append your section here when done.*
