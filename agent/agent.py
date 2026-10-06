"""
agent/agent.py  —  Part 3 will implement this file.

Part 3 must implement:
    class StudyAgent:
        __init__(self, index: NotesIndex)
        run(self, user_message: str) -> dict

    run() return format:
        {
            "task": "qa|summary|quiz|topics|flashcards|calculate|web",
            "output": str | list[dict] | list[str],
            "sources": list[dict],
            "tool_used": str
        }

See CONTRACT.md for routing rules and field details.
"""

from agent.retriever import NotesIndex


class StudyAgent:
    """Placeholder class — Part 3 will implement this. See CONTRACT.md."""

    def __init__(self, index: NotesIndex):
        raise NotImplementedError("Part 3 must implement StudyAgent.__init__")

    def run(self, user_message: str) -> dict:
        raise NotImplementedError("Part 3 must implement StudyAgent.run")
