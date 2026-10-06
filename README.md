# Study Agent 🎓

Study Agent is a college project that turns PDF notes into a study workspace.
Upload a PDF, ask questions with page references, generate a summary, take a
multiple-choice quiz, tick off important topics, or reveal flashcard answers.
The interface shows which tool the agent chose and the steps it actually ran.

**Status: Parts 1, 2, 3 and 4 are complete.** Part 4 delivered the Streamlit UI,
interactive study controls, agent traces, tests and presentation documentation.

## Architecture

```text
                         Streamlit (app.py)
                       /                  \
             Upload + Process         Chat / quick actions
                    |                         |
               load_pdf()                     |
                    |                         |
              chunk_pages()                   |
                    |                         |
               NotesIndex ----------------> StudyAgent.run()
           TF-IDF, in session_state           |
                                      Gemini tool selection
                                      (maximum five turns)
                                               |
                +------------------------------+-------------------+
                |                  |                 |              |
          search_notes        calculator        web_search     generators
          NotesIndex          safe AST          optional DDGS  summary/quiz/
                |                                              topics/cards
                +------------------------------+-------------------+
                                               |
                                  task + output + sources + steps
                                               |
                                      Streamlit renderers
                               chat / quiz / cards / download

If function calling fails: keyword router -> existing tools / answer_question
Text generation: ask_llm() -> Gemini (google-genai SDK)
```

The UI calls `StudyAgent.run()` for every study request. It only handles PDF
processing, session state, presentation and quiz scoring; it does not route
requests itself. See [CONTRACT.md](CONTRACT.md) for public interfaces and
[HANDOFF.md](HANDOFF.md) for exact output shapes and changes across all four parts.

## Setup and run

Use Python 3.10 or newer. From a terminal:

```bash
git clone https://github.com/Chetanprasanna/study-agent.git
cd study-agent
python -m venv venv
source venv/bin/activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create your local configuration, if you do not already have one:

```bash
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
```

Edit `.env`, replacing the placeholder with your Gemini key:

```dotenv
GEMINI_API_KEY=your_key_here
```

Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).
The repository's configured model is `gemini-3.5-flash-lite`, defined once in
`agent/llm.py`. Your account must have access to the configured model.
Keep `.env` local; it is ignored by Git.

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.
If `GEMINI_API_KEY` is absent, enter a key in the sidebar's password field.
That fallback stays in the browser session's server-side memory and is passed
through a request-scoped context; it is never written to `.env` or `os.environ`.
An environment key takes priority.

1. Upload a PDF containing selectable text, then click **Process notes**.
2. Wait for the ready message showing text pages and chunks.
3. Type a question or click **Summary**, **Quiz**, **Important topics**, or **Flashcards**.
4. Expand **Sources (pages)** and **Agent steps** to inspect the reply.
5. For a quiz, select all answers and click **Check answers**. The score and
   explanations persist across reruns; submit again after changing choices.
6. Reveal flashcards with their expanders, tick topics as you revise, and use
   **Download summary (.txt)** to save a summary.

Processing happens locally; answering and generation send selected note excerpts
and your request to Gemini. Chat, index and study controls remain in session
memory. Changing/removing the upload clears document-specific history and quiz
answers. The **Agent tools** panel tracks tools attempted throughout the session,
including failed attempts. Refreshing/disconnecting can end the session.

Optional web search:

```bash
python -m pip install ddgs
```

DDGS does not need a search API key in this implementation. When not installed,
the backend's disabled message mentions a search API key for contract compatibility;
installing `ddgs` is the actual enablement step.

## Example prompts

Use questions matching your own PDF's subject. For biology notes, try:

- “What is ATP? Cite the relevant pages.”
- “Summarize my notes.”
- “Make 5 quiz questions from my notes.”
- “List the important topics in my notes.”
- “Create 8 flashcards from my notes.”
- “Calculate (12 + 8) / 4.”
- “Find ATP in my notes, then calculate 12 * 8.”
- “Search the web for an explanation of cellular respiration.”

## How the agent decides

Gemini receives the student's message and descriptions of seven available tools:
`search_notes`, `calculator`, `web_search`, `summarize`, `make_quiz`,
`important_topics`, and `make_flashcards`. It can call multiple tools, inspect
results, and return a final answer. The loop stops after at most five model turns.
The sidebar calls `make_quiz` **Generate Quiz**; the reply badge shows its actual
backend name.

If tool calling fails, the agent records `fallback_router` and uses keyword
matching to select one existing tool. Fallback Q&A uses `answer_question()`.
The UI does not claim fallback routing was a successful Gemini decision:
its trace shows the fallback event and recorded error.

Quizzes, topics and cards keep their structured tool outputs. For successful
multi-tool requests, the last non-search tool determines the displayed task;
earlier results remain visible in **Agent steps**. Steps are observable tool
inputs and outputs, not hidden model reasoning. The panel marks actual recorded
tool attempts rather than assuming the reply's `tool_used` field proves execution.

## Tests and command-line tools

```bash
python -m pytest tests/ -q
python -m pytest tests/test_part4.py -v
python scripts/try_agent.py path/to/notes.pdf
python scripts/try_qa.py path/to/notes.pdf "What is the main idea?"
```

Part 4 tests generate small real PDFs in memory, build a real TF-IDF index, and
exercise both fallback Q&A and Gemini's tool loop with model responses mocked.
Streamlit AppTest covers missing inputs, errors, chat, quiz grading, output types,
API-key fallback, session caching, PDF replacement and removal. No live Gemini
or web requests are needed. The optional Part 1 real-PDF test skips when no PDF
is present in `samples/`.

## Project files

```text
app.py                 Streamlit UI and importable presentation helpers
agent/pdf_loader.py    PDF -> pages
agent/chunker.py       Pages -> overlapping chunks
agent/retriever.py     TF-IDF search
agent/llm.py           Gemini text calls and scoped API-key helpers
agent/qa.py            Retrieval-augmented Q&A
agent/tools.py         Search, safe calculator, optional web search
agent/generators.py    Summary, validated quizzes, topics, flashcards
agent/agent.py         Function-calling loop and fallback router
tests/test_part1.py    PDF/chunking tests
tests/test_part2.py    Retrieval/LLM/Q&A tests
tests/test_part3.py    Tools/generators/agent tests
tests/test_part4.py    UI, pipeline and regression tests
scripts/               Interactive CLI entry points
samples/               Optional local sample PDFs
DEMO.md                Three-minute college presentation script
CONTRACT.md            Stable backend interfaces
HANDOFF.md             Completion log and implementation details
```

## Limitations

- Scanned/image-only PDFs need OCR before uploading; OCR is not built in.
- One PDF is active at a time. Blank pages are skipped; citations retain original
  one-based PDF page positions, which may differ from printed page labels.
- TF-IDF matches words, not meaning, so synonyms and unusual phrasing can miss notes.
- Generators sample up to 12 evenly spaced excerpts, each capped at 1,200 characters.
  Large PDFs can have details omitted; PDF parsing/indexing still runs in memory.
- Gemini can make mistakes. Verify the cited passages and generated answer keys.
  Network access, quotas, credentials and model availability affect requests.
- Each request starts a fresh backend conversation. Chat history is displayed but
  is not sent back to the agent, so follow-up questions should be self-contained.
- Fallback routing selects one task and cannot reproduce multi-tool planning.
- Web search depends on optional DDGS and its external service availability.
- No permanent chat storage or user accounts. Terminal traces include prompts
  and note excerpts; this is a local educational app, not a hardened hosted service.

## Future ideas

- Semantic embeddings and a vector database for better retrieval.
- OCR for scanned notes and improved extraction of tables and equations.
- Multi-PDF libraries with document names in citations.
- Optional conversational memory and saved revision progress.
- Spaced repetition, exportable quizzes and model-selection settings.

For the presentation, follow [DEMO.md](DEMO.md).
