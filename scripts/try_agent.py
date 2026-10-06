"""Load a PDF and chat with Study Agent: python scripts/try_agent.py notes.pdf."""

import argparse
import json
import sys
from pathlib import Path

# Direct script execution starts inside scripts/, so add the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import StudyAgent
from agent.chunker import chunk_pages
from agent.pdf_loader import load_pdf
from agent.retriever import NotesIndex


def main() -> None:
    """Index one PDF, then show responses and the agent's printed tool steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the PDF notes")
    args = parser.parse_args()
    if not args.pdf.is_file() or args.pdf.suffix.lower() != ".pdf":
        parser.error("Provide an existing PDF file.")
    try:
        pages = load_pdf(args.pdf)
        if not pages:
            parser.error("The PDF has no readable text.")
        chunks = chunk_pages(pages)
        agent = StudyAgent(NotesIndex(chunks))
    except Exception as exc:
        parser.exit(1, f"Could not load notes: {exc}\n")
    print(f"Loaded {len(pages)} pages, {len(chunks)} chunks. Type quit to exit.")
    print("Try: summarize notes | make 5 quiz questions | calculate (12 + 8) / 4")
    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if message.lower() in {"quit", "exit", "q"}:
            break
        if not message:
            continue
        result = agent.run(message)
        output = result["output"]
        print(f"\nAgent ({result['task']}):")
        print(json.dumps(output, indent=2, ensure_ascii=False)
              if isinstance(output, list) else output)
        # The run method already prints the trace; list page numbers separately.
        pages_used = sorted({source["page"] for source in result["sources"]})
        if pages_used:
            print(f"Sources: pages {pages_used}")


if __name__ == "__main__":
    main()
