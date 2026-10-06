"""
agent/generators.py  —  Part 3 will implement this file.

Part 3 must implement:
    summarize(index: NotesIndex) -> str
    make_quiz(index: NotesIndex, n: int = 5) -> list[dict]
        Each quiz item: {"question", "options": [4 str], "answer_index": int, "explanation"}
    important_topics(index: NotesIndex) -> list[str]
    make_flashcards(index: NotesIndex, n: int = 8) -> list[dict]
        Each flashcard: {"front": str, "back": str}

See CONTRACT.md for full requirements and return formats.
"""

from agent.retriever import NotesIndex


def summarize(index: NotesIndex) -> str:
    """Placeholder — Part 3 will implement this. See CONTRACT.md."""
    raise NotImplementedError("Part 3 must implement summarize")


def make_quiz(index: NotesIndex, n: int = 5) -> list[dict]:
    """Placeholder — Part 3 will implement this. See CONTRACT.md."""
    raise NotImplementedError("Part 3 must implement make_quiz")


def important_topics(index: NotesIndex) -> list[str]:
    """Placeholder — Part 3 will implement this. See CONTRACT.md."""
    raise NotImplementedError("Part 3 must implement important_topics")


def make_flashcards(index: NotesIndex, n: int = 8) -> list[dict]:
    """Placeholder — Part 3 will implement this. See CONTRACT.md."""
    raise NotImplementedError("Part 3 must implement make_flashcards")
