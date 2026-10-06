"""
agent/tools.py  —  Part 3 will implement this file.

Part 3 must implement:
    search_notes(query: str, index: NotesIndex) -> list[dict]
    calculator(expression: str) -> str   # safe eval using ast, NO eval()/exec()
    web_search(query: str) -> str        # may return "not enabled" message

See CONTRACT.md for full requirements and safety rules.
"""

from agent.retriever import NotesIndex


def search_notes(query: str, index: NotesIndex) -> list[dict]:
    """Placeholder — Part 3 will implement this. See CONTRACT.md."""
    raise NotImplementedError("Part 3 must implement search_notes")


def calculator(expression: str) -> str:
    """Placeholder — Part 3 will implement this. See CONTRACT.md."""
    raise NotImplementedError("Part 3 must implement calculator")


def web_search(query: str) -> str:
    """Placeholder — Part 3 will implement this. See CONTRACT.md."""
    raise NotImplementedError("Part 3 must implement web_search")
