# Study Agent — API Contract

**Project status: Parts 1, 2, 3 and 4 are complete.** The Streamlit UI is
implemented; this contract remains the reference for maintenance and future changes.

> **This document is LAW.**  
> Every function signature, return type, and field name listed here is binding.  
> All completed parts and future changes must preserve these interfaces — no renames, no type changes, no missing fields.
> If you need to add a helper function, that's fine. But the public API below cannot be altered.

---

## Table of Contents

1. [agent/pdf_loader.py](#1-agentpdf_loaderpy)
2. [agent/chunker.py](#2-agentchunkerpy)
3. [agent/retriever.py](#3-agentretrieverpy)
4. [agent/llm.py](#4-agentllmpy)
5. [agent/qa.py](#5-agentqapy)
6. [agent/generators.py](#6-agentgeneratorspy)
7. [agent/tools.py](#7-agenttoolspy)
8. [agent/agent.py](#8-agentagentpy)
9. [Code Rules — All Parts Must Follow](#9-code-rules--all-parts-must-follow)

---

## 1. `agent/pdf_loader.py`

```python
def load_pdf(file) -> list[dict]:
    ...
```

- **`file`**: a file path (string or `Path`) **or** a file-like object (e.g. Streamlit's `UploadedFile`)
- **Returns**: a list of page dicts, one per non-empty page:
  ```python
  [
      {"page": 1, "text": "Full text of page 1..."},
      {"page": 3, "text": "Full text of page 3..."},  # page 2 was blank, skipped
      ...
  ]
  ```
- **Skips** pages where the extracted text (stripped) is empty.
- Page numbers are **1-indexed**.

---

## 2. `agent/chunker.py`

```python
def chunk_pages(pages, chunk_size=800, overlap=150) -> list[dict]:
    ...
```

- **`pages`**: the list returned by `load_pdf()`
- **`chunk_size`**: target number of characters per chunk
- **`overlap`**: number of characters to repeat at the start of each new chunk (for context continuity)
- **Returns**: a list of chunk dicts:
  ```python
  [
      {"id": 0, "page": 1, "text": "First chunk text..."},
      {"id": 1, "page": 1, "text": "Overlapping second chunk..."},
      {"id": 2, "page": 2, "text": "First chunk from page 2..."},
      ...
  ]
  ```
- `id` is a global integer index starting at 0.
- Each chunk carries the page number it originated from.
- Chunking should prefer splitting on paragraph (`\n\n`) or sentence (`. `) boundaries where possible.

---

## 3. `agent/retriever.py`

```python
class NotesIndex:
    def __init__(self, chunks: list[dict]):
        ...

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        ...
```

- **`__init__`**: accepts the list returned by `chunk_pages()` and builds a TF-IDF index (using `sklearn`).
- **`search`**: returns the top-k most relevant chunks for a query.
- **Return format** (each item is a chunk dict with an extra field):
  ```python
  [
      {"id": 0, "page": 1, "text": "...", "score": 0.87},
      ...
  ]
  ```
- Results should be sorted by `score` descending.

---

## 4. `agent/llm.py`

```python
def ask_llm(prompt: str, system: str | None = None) -> str:
    ...
```

- Calls the **Google Gemini API** using the `google-genai` SDK.
- Reads the API key from the environment variable `GEMINI_API_KEY` (loaded via `python-dotenv`).
- **`system`**: optional system instruction string; pass it to the model if provided.
- Returns the model's text response as a plain string.
- Must **not crash on import** if `GEMINI_API_KEY` is missing — only raise when actually called.

---

## 5. `agent/qa.py`

```python
def answer_question(question: str, index: NotesIndex) -> dict:
    ...
```

- Uses `index.search()` to retrieve relevant chunks.
- Builds a prompt from the retrieved chunks and calls `ask_llm()`.
- **Returns**:
  ```python
  {
      "answer": "The mitochondria is...",
      "sources": [
          {"id": 2, "page": 3, "text": "...", "score": 0.91},
          ...
      ]
  }
  ```
- `sources` is the raw list returned by `index.search()`.

---

## 6. `agent/generators.py`

```python
def summarize(index: NotesIndex) -> str:
    ...

def make_quiz(index: NotesIndex, n: int = 5) -> list[dict]:
    ...

def important_topics(index: NotesIndex) -> list[str]:
    ...

def make_flashcards(index: NotesIndex, n: int = 8) -> list[dict]:
    ...
```

### `summarize`
- Returns a plain string: a concise summary of the entire document.

### `make_quiz`
- Returns a list of `n` quiz questions:
  ```python
  [
      {
          "question": "What is photosynthesis?",
          "options": ["Option A", "Option B", "Option C", "Option D"],  # exactly 4 strings
          "answer_index": 2,   # 0-based index into options
          "explanation": "Because..."
      },
      ...
  ]
  ```

### `important_topics`
- Returns a list of strings, each a key topic from the notes.

### `make_flashcards`
- Returns a list of `n` flashcard dicts:
  ```python
  [
      {"front": "What is ATP?", "back": "Adenosine triphosphate, the energy currency of the cell."},
      ...
  ]
  ```

---

## 7. `agent/tools.py`

```python
def search_notes(query: str, index: NotesIndex) -> list[dict]:
    ...

def calculator(expression: str) -> str:
    ...

def web_search(query: str) -> str:
    ...
```

### `search_notes`
- Thin wrapper around `index.search(query)`.
- Returns the same format as `NotesIndex.search()`.

### `calculator`
- Evaluates a **safe** mathematical expression string.
- Must **not** use Python's `eval()` or `exec()` on arbitrary code.
- Use `ast` + `operator` or a library like `simpleeval`.
- Returns the result as a string (e.g. `"42"` or `"3.14"`).
- On error, returns a descriptive error string (not a raised exception).

### `web_search`
- Optionally performs a real web search.
- If not enabled/configured, returns the string:  
  `"Web search is not enabled. Please enable it by configuring a search API key."`
- Return type is always `str`.

---

## 8. `agent/agent.py`

```python
class StudyAgent:
    def __init__(self, index: NotesIndex):
        ...

    def run(self, user_message: str) -> dict:
        ...
```

### `run` return format

```python
{
    "task": "qa",           # one of: "qa" | "summary" | "quiz" | "topics" | "flashcards" | "calculate" | "web"
    "output": ...,          # str for qa/summary/topics/calculate/web; list[dict] for quiz/flashcards; list[str] for topics
    "sources": [...],       # list of source chunk dicts (empty list [] if not applicable)
    "tool_used": "search_notes"  # name of the tool/function used
}
```

- The agent must classify the user's intent from `user_message` and route to the correct function.
- For `"topics"`, `output` is `list[str]`.
- For `"quiz"`, `output` is `list[dict]` (quiz question dicts).
- For `"flashcards"`, `output` is `list[dict]` (flashcard dicts).
- For all others, `output` is `str`.

---

## 9. Code Rules — All Parts Must Follow

These rules apply to **every file** in the `agent/` directory.

### ✅ Must Do

1. **Type hints on every function** — parameters and return types.
2. **Docstring on every function** — at minimum a one-liner explaining what it does.
3. **Heavy inline comments** — this is a beginner-friendly educational project. Comment the "why", not just the "what". Assume the reader is a first-year CS student.
4. **Use `python-dotenv`** to load environment variables. Call `load_dotenv()` at module level.
5. **Every module must be importable without crashing** even if `GEMINI_API_KEY` is missing from the environment. The key should only be read/validated inside the function that needs it (e.g. `ask_llm()`).
6. **No hardcoded API keys** — ever. Use `.env` only.
7. **Pytest tests** — each part writes `tests/test_part<N>.py` covering their modules.
8. **Append to `HANDOFF.md`** — when your part is done, add a section describing what you built, what the next part must know, and how to run your tests.

### ❌ Must Not Do

1. Do not rename, remove, or change the signature of any function listed in this contract.
2. Do not use `eval()` or `exec()` for the calculator — use `ast`-based safe evaluation.
3. Do not hardcode any string that should come from the model (e.g. do not fake LLM responses in production code).
4. Do not import from a sibling module in a way that causes circular imports.
5. Do not break existing tests when adding new code.

---

*Contract written by Part 1. Version: 1.0.0*
