"""
scripts/try_qa.py  —  Part 2 CLI tool

End-to-end test script: load a PDF → chunk it → build index → answer a question.
Use this to verify the full pipeline works before the Streamlit UI (Part 4) exists.

Usage
-----
    # From the repo root, with venv activated:
    python scripts/try_qa.py path/to/notes.pdf "What is photosynthesis?"

    # Ask multiple questions interactively (omit the question argument):
    python scripts/try_qa.py path/to/notes.pdf

Requirements
------------
    - GEMINI_API_KEY must be set in .env (or exported in your shell)
    - pip install -r requirements.txt

Exit codes
----------
    0  success
    1  bad arguments or file not found
    2  missing API key
    3  API error
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys       # sys.argv for command-line arguments, sys.exit for exit codes
import textwrap  # textwrap.fill() to wrap long lines for terminal display
from pathlib import Path

# ── Local imports ─────────────────────────────────────────────────────────────
# Add the repo root to sys.path so we can import `agent` regardless of where
# the script is called from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pdf_loader import load_pdf
from agent.chunker import chunk_pages
from agent.retriever import NotesIndex
from agent.qa import answer_question
from agent.llm import MissingApiKeyError


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_separator(char: str = "─", width: int = 70) -> None:
    """Print a horizontal rule to visually separate sections in the terminal."""
    print(char * width)


def print_answer(result: dict) -> None:
    """
    Pretty-print the result dict from answer_question().

    Parameters
    ----------
    result : dict
        Must have keys "answer" (str) and "sources" (list[dict]).
    """
    print_separator()
    print("📖  ANSWER:")
    print_separator()

    # Wrap the answer text at 70 chars for readability in narrow terminals.
    wrapped = textwrap.fill(result["answer"], width=70)
    print(wrapped)
    print()

    # Print sources so the user can verify which parts of the PDF were used.
    if result["sources"]:
        print("📌  SOURCES USED:")
        for i, src in enumerate(result["sources"], start=1):
            # Show page number and relevance score, plus a snippet of the text.
            snippet = src["text"][:120].replace("\n", " ")
            if len(src["text"]) > 120:
                snippet += "..."
            print(f"  [{i}] Page {src['page']} (score: {src['score']:.3f})")
            print(f"      \"{snippet}\"")
    else:
        print("  (No sources — question may be outside the notes' scope)")

    print_separator()


def load_and_index(pdf_path: Path) -> tuple[NotesIndex, int]:
    """
    Load a PDF and build a NotesIndex from it.

    Parameters
    ----------
    pdf_path : Path
        Absolute or relative path to the PDF file.

    Returns
    -------
    tuple[NotesIndex, int]
        (index, number_of_chunks)

    Raises
    ------
    FileNotFoundError
        If the PDF path does not exist.
    """
    print(f"📄  Loading PDF: {pdf_path}")

    # ── Step 1: Load pages from PDF ───────────────────────────────────────────
    pages = load_pdf(pdf_path)
    print(f"    ✓ {len(pages)} non-empty pages found")

    if not pages:
        print("⚠️   The PDF appears to be blank or unreadable. Exiting.")
        sys.exit(1)

    # ── Step 2: Chunk the pages ───────────────────────────────────────────────
    chunks = chunk_pages(pages, chunk_size=800, overlap=150)
    print(f"    ✓ {len(chunks)} chunks created (chunk_size=800, overlap=150)")

    # ── Step 3: Build the TF-IDF index ───────────────────────────────────────
    print("🔍  Building search index...")
    index = NotesIndex(chunks)
    print(f"    ✓ Index ready ({len(chunks)} chunks indexed)")
    print()

    return index, len(chunks)


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """
    Parse command-line arguments, load the PDF, and run Q&A.

    Usage:
        python scripts/try_qa.py <pdf_path> [question]

    If [question] is omitted, enters an interactive loop.
    """

    # ── Argument validation ───────────────────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python scripts/try_qa.py <pdf_path> [question]")
        print("Example: python scripts/try_qa.py notes/biology.pdf 'What is ATP?'")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"❌  File not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print(f"❌  Expected a .pdf file, got: {pdf_path.suffix}")
        sys.exit(1)

    # ── Load and index the PDF ────────────────────────────────────────────────
    try:
        index, num_chunks = load_and_index(pdf_path)
    except Exception as exc:
        print(f"❌  Failed to load PDF: {exc}")
        sys.exit(1)

    # ── Run Q&A ───────────────────────────────────────────────────────────────
    if len(sys.argv) >= 3:
        # A question was provided on the command line — answer it and exit.
        question = sys.argv[2]
        print(f"❓  Question: {question}")
        print()

        try:
            result = answer_question(question, index)
            print_answer(result)

        except MissingApiKeyError as exc:
            print(f"\n❌  API Key Error: {exc}")
            sys.exit(2)

        except RuntimeError as exc:
            print(f"\n❌  API Error: {exc}")
            sys.exit(3)

    else:
        # No question provided — run an interactive loop.
        print("💬  Interactive mode. Type 'quit' or press Ctrl+C to exit.")
        print(f"    Index contains {num_chunks} chunks from {pdf_path.name}.")
        print()

        while True:
            try:
                question = input("❓  Your question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋  Goodbye!")
                break

            if not question:
                print("    (Please enter a question)")
                continue

            if question.lower() in {"quit", "exit", "q", "bye"}:
                print("👋  Goodbye!")
                break

            try:
                result = answer_question(question, index)
                print_answer(result)

            except MissingApiKeyError as exc:
                print(f"\n❌  API Key Error: {exc}")
                sys.exit(2)

            except RuntimeError as exc:
                print(f"\n❌  API Error: {exc}")
                # Don't exit on API errors in interactive mode — let the user retry.
                print("    (You can try asking a different question)")
                print()


if __name__ == "__main__":
    main()
