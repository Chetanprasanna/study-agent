# Study Agent — Handoff Log

Each part appends a new section to this file when complete.  
All four parts are now complete. Read the previous sections for implementation
history and the Part 4 DONE section for the final checklist and run instructions.

**Current status: All four parts are complete, including the Streamlit UI.**
**Current shared model:** `gemini-3.5-flash-lite`.

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

### Part 2 Handoff to Part 3 (Historical — Part 3 Is Now Complete)

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

### Part 2 Limitations Recorded Before Part 3

- TF-IDF retrieval is keyword-based, not semantic. If the user phrases their question
  very differently from the notes, retrieval may miss relevant chunks. Part 3's agent
  could re-phrase the query before searching to mitigate this.
- `ask_llm` currently has no token-count guard. For very large PDFs, the prompt 
  (context + question) could exceed Gemini's context window. Part 3 generators should
  limit context to a reasonable number of chunks (e.g. top-8 for summarisation).
- `gemini-2.5-flash` is the model used. If it's deprecated by the time you build Part 3,
  update `_DEFAULT_MODEL` in `agent/llm.py`.

---

## Part 3 DONE — Generators, Tools & Gemini Agent Loop ✅

**Date:** 2026-10-06
**Files:** `agent/generators.py`, `agent/tools.py`, `agent/agent.py`,
`tests/test_part3.py`, `scripts/try_agent.py`, `requirements.txt`, `HANDOFF.md`.

### What was built

- Summary, MCQ quiz, important topics and flashcards reuse Part 2's `ask_llm`.
  Broad tasks sample up to 12 evenly spaced chunks, including the beginning
  and end. Each prompt excerpt is capped at 1,200 characters. This gives bounded
  document coverage, but can omit details in very large PDFs.
- JSON generators remove enclosing code fences, require an array, validate every
  field and exact quiz/card counts, and retry once after bad syntax or schema.
  API failures and exhausted JSON retries return the error shapes below.
- `search_notes` wraps `NotesIndex.search` unchanged. The calculator walks an
  AST, allows numbers, unary signs, + - * / ** %, and parentheses, and rejects
  names, function calls, attributes, containers, booleans and unsupported
  operators. Expression length, tree size, powers and result size are bounded.
- Optional web search uses `ddgs` only when installed, returns at most three
  results, and catches network errors. The optional requirement is commented
  out. Install with `pip install ddgs`; DDGS itself needs no search API key.
  The missing-package message retains the exact wording required by CONTRACT.md.
- `StudyAgent.run` declares all seven tools and manually runs at most five
  Gemini turns using the same model constant as Part 2. Each turn sends the
  conversation and declarations, executes requested tools, appends function
  responses, and continues until a text response. Parallel tool calls receive
  one combined response turn; call IDs and original model parts are preserved.
  Tool argument errors are returned to Gemini so it can correct them.
- The keyword router runs only if function calling fails, including missing
  credentials, empty model responses, SDK/API errors or iteration exhaustion.
  Its Q&A path reuses `answer_question`; normal agent Q&A uses the model's
  final text with chunks gathered by `search_notes`.
- Every tool execution and final response is stored and printed. The trace
  shows observable actions, inputs and results, not private model reasoning.
- The CLI loads a PDF once and accepts messages until quit/exit or Ctrl+C.
  Each message starts a fresh conversation over the same index.

### Exact output shapes for Part 4

Every `StudyAgent.run(message)` returns:

```python
{
    "task": "qa" | "summary" | "quiz" | "topics" | "flashcards" | "calculate" | "web",
    "output": str | list[dict] | list[str],
    "sources": list[dict],
    "tool_used": str,
    "steps": [
        {"thought_or_tool": str, "input": str | dict, "observation": str | list | dict}
    ],
}
```

Successful generator outputs:

```python
summarize(index)  # str: concise summary with page references
make_quiz(index, n=5)  # exactly n objects:
[
    {
        "question": str,
        "options": [str, str, str, str],
        "answer_index": int,  # 0..3; bool is rejected
        "explanation": str,
    }
]
important_topics(index)  # list[str]: non-empty topic names
make_flashcards(index, n=8)  # exactly n objects:
[{"front": str, "back": str}]
```

Counts must be integers from 1 to 50. Quiz/card generation errors return
`[{"error": "Error: ..."}]`; topics errors return `["Error: ..."]`;
summary errors return `"Error: ..."`. Check these sentinels before rendering
quiz options or card faces. These are failure payloads, not valid study items.
They keep the public list return types without fabricating study content.

- `qa`: `output` is a string; `tool_used` is `"search_notes"`.
- `summary`: string; `tool_used` is `"summarize"`.
- `quiz`: list of the MCQ dicts above (or error sentinel);
  `tool_used` is `"make_quiz"`.
- `topics`: list of strings (or error sentinel);
  `tool_used` is `"important_topics"`.
- `flashcards`: list of card dicts above (or error sentinel);
  `tool_used` is `"make_flashcards"`.
- `calculate`: string; `tool_used` is `"calculator"`.
- `web`: string containing titles, snippets and URLs, or a disabled/error
  message; `tool_used` is `"web_search"`.

Structured quiz/card/topic outputs are preserved even if Gemini's closing
response is prose. For a successful multi-tool request, the last non-search
tool determines the task; all earlier outputs remain available in `steps`.
For string tasks the agent uses Gemini's closing text; fallback uses the raw
tool output. The router does not attempt to reproduce multi-tool planning.

Retrieved source dicts are `{"id": int, "page": int, "text": str, "score": float}`.
Sampled generator sources are `{"id": int, "page": int, "text": str}` without
a relevance score. The UI should use `source.get("score")` if showing scores.
Sources are empty when not applicable; a multi-tool run can retain earlier
retrieved sources. Step inputs are argument dicts for tools and message strings
for fallback/final events. Full observations are stored; terminal previews are
capped at 600 characters.

### How to test and demo

From the repository root, with the existing virtual environment:

```bash
source venv/bin/activate
python -m pytest tests/test_part3.py -v
python -m pytest tests/ -q
python scripts/try_agent.py samples/my_notes.pdf
# Optional web support:
pip install ddgs
```

Set `GEMINI_API_KEY` in `.env` to demonstrate real Gemini tool selection.
Try "Find ATP in my notes, then calculate 12 * 8", "Make 5 quiz questions",
"Summarize my notes", or "Create 8 flashcards".

Validation: 61 Part 3 tests passed. The latest local full-suite run passed
131 tests; the optional sample-PDF test skips when `samples/my_notes.pdf` is absent. Tests mock `ask_llm` and Gemini SDK responses,
including multi-turn and parallel calls, argument recovery, malformed JSON,
missing credentials and the five-turn limit. Imports and the CLI work without
credentials. Live Gemini and web requests were not exercised.

### Changes to earlier parts

The Part 3 implementation kept Parts 1 and 2 files and CONTRACT.md unchanged.
The subsequent user-requested model update changed `agent/llm.py` as recorded below.

## Part 3 configuration update — 2026-10-06

- At the user's request, changed Part 2's shared model constant in `agent/llm.py` to `gemini-3.5-flash-lite`. Q&A, generators and the function-calling agent now all use this model. Public signatures are unchanged.
- Credentials are stored only in the local, untracked `.env`, with owner-only file permissions. `.gitignore` also excludes `.env.*` variants while keeping the placeholder-only `.env.example` trackable.


## Part 4 — Original UI Handoff Requirements Completed

The UI handoff from Part 3 has been fulfilled:

- [x] Built the UI against CONTRACT.md and the Part 3 output shapes above.
- [x] Reused the PDF pipeline, `NotesIndex` and `StudyAgent.run`.
- [x] Rendered text, quiz/card/topic lists, generation errors, sources and steps.
- [x] Kept `.env` credentials local and ignored by Git.
- [x] Added `tests/test_part4.py` and the completed Part 4 handoff below.

## Part 4 DONE — Streamlit Study Workspace ✅

**Date:** 2026-10-06

**Files:** `app.py`, `tests/test_part4.py`, `README.md`, `DEMO.md`,
`HANDOFF.md`, `agent/llm.py`, `agent/agent.py`, `agent/retriever.py`,
`CONTRACT.md`, `scripts/try_qa.py`.

### Final checklist

- [x] Sidebar PDF uploader and Process notes button reuse `load_pdf` ->
  `chunk_pages` -> `NotesIndex`; pages, chunks and index live in session state.
- [x] Environment/.env key takes priority; a password input supplies the fallback
  through a request-scoped context without mutating process environment variables.
- [x] All four quick actions and chat call `StudyAgent.run()` only.
- [x] Chat history and decision badges display the backend task and real tool name.
- [x] Q&A displays expandable source pages and excerpts; relevance scores are
  optional because sampled generator sources do not include them.
- [x] Quizzes use radio choices, Check answers, a persistent score and explanations.
  Incomplete submissions are rejected. Each reply has independent widget keys.
- [x] Flashcard expanders reveal answers, topic checkboxes track revision, and
  summaries provide a `.txt` download.
- [x] Each reply has an Agent steps expander with the recorded sequence and
  full input/observation data. Fallback events remain visible.
- [x] Agent tools lists Search Notes, Calculator, Web Search, Generate Quiz and
  the other generators, marking tools attempted during the session.
- [x] Missing uploads/keys, unreadable or corrupt PDFs, API failures and all
  documented generator error sentinels display friendly messages.
- [x] Changed/removed PDFs clear stale index, chat and study-widget state.
  The session-wide tool-usage set intentionally survives document changes.
- [x] README includes architecture, setup, prompts, routing, limitations and ideas.
- [x] DEMO.md provides a timed three-minute presentation and recovery lines.
- [x] Offline pipeline, UI interaction and regression tests pass.

### Integration details

The UI's helpers can be imported without starting Streamlit because `main()` is
guarded by `if __name__ == "__main__"`. `build_notes()` builds the existing PDF
pipeline; `run_request()` calls the existing agent under `use_api_key()`.
`output_error()` checks string, list-of-string and list-of-error-dict sentinels
before rendering. `score_quiz()` only grades returned answers; it never generates
content. `tools_used()` reads actual trace events (including fallback
`answer_question`) instead of assuming a `tool_used` label proves execution.

The badge displays `make_quiz`, matching the backend contract; the sidebar label
is Generate Quiz. Successful multi-tool replies follow Part 3's existing rule:
the last non-search tool controls the main output, with earlier results in steps.
The UI adds a final-answer label when fallback returns without a final-response
event; it does not invent additional tool calls or expose private reasoning.

Session widget keys start with `reply_` and include the message position. Forms
batch quiz input. Submitted answers are stored separately from the live choices,
so the displayed score describes the last checked answers until resubmission.
Widget reruns and downloads do not issue a fresh agent request.

### Changes and fixes to earlier parts

1. **Part 2/3 credential integration:** `agent/llm.py` adds `get_api_key()` and
   `use_api_key()` using `ContextVar`; both `ask_llm()` and `StudyAgent.run()` use
   the shared resolver. Environment configuration remains the default and takes
   priority. The fallback context resets even on exceptions and is isolated from
   other threads. Existing public signatures and model selection are unchanged.
2. **Part 2 repeated-document indexing bug:** for at least 20 highly repetitive
   chunks, `max_df=0.95` could remove every term and raise during indexing.
   `NotesIndex` now retries with `max_df=1.0` only for that specific pruning error.
   Empty-vocabulary errors still reach the UI as unreadable-word validation.
3. **Part 2 ownership bug:** the index claimed to copy input chunks but retained
   their dictionary references. It now copies each chunk so caller edits cannot
   make displayed source text disagree with the fitted vectors.
4. **Part 2 search-limit bug:** negative `top_k` previously used negative slicing
   and returned results. Nonpositive limits now return an empty list.

All three retriever fixes have regression coverage in `tests/test_part4.py`.
The public API contracts and all existing tests remain unchanged.

### Validation and run instructions

From the repository root:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
# First-time setup only; do not overwrite an existing .env:
cp .env.example .env
# Edit .env and set GEMINI_API_KEY, or use the sidebar fallback.
streamlit run app.py

# In a second terminal with the same virtual environment activated:
python -m pytest tests/test_part4.py -v
python -m pytest tests/ -q
```

Validation on 2026-10-06: **24 Part 4 tests passed; 155 tests passed overall**.
The run used Python 3.14 and Streamlit 1.65.0. There is one upstream google-genai
Python 3.14 deprecation warning. Tests generate real PDF bytes and exercise actual
parsing, indexing and agent execution with mocked `ask_llm`/Gemini responses.
AppTest covers quiz scoring, chat, quick actions, all output/error shapes, sources,
steps, caching, API-key fallback, upload replacement/removal and invalid PDFs.
AppTest substitutes the uploader's byte transport because it has no upload setter.
Live Gemini calls, live web search and a manual browser/download check were not
performed. The existing local sample PDF remains untracked; it is not required
for Part 4 tests and is not included in the suggested commit commands.

Chat history is presentation-only: the existing agent still starts a fresh
conversation for every request. No persistence, OCR or multi-PDF support was added.

### Final completion-status review — 2026-10-06

- README.md, CONTRACT.md and DEMO.md explicitly mark all four parts complete.
- The original Part 4 handoff is now a fulfilled checklist, with no pending UI work.
- The `app.py` module description identifies the completed Part 4 UI.
- `scripts/try_qa.py` now describes the CLI as a companion to the completed UI;
  its obsolete claim that the Streamlit UI does not yet exist was removed.
- Earlier part logs remain implementation history. Contract status wording was
  updated without changing any public interface or application behavior.
