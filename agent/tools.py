"""Small tools the agent can call; arithmetic never executes Python code."""

import ast
import math
import operator
from typing import Callable

from dotenv import load_dotenv

from agent.retriever import NotesIndex

load_dotenv()
_BINARY: dict[type, Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY: dict[type, Callable] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_DISABLED = (
    "Web search is not enabled. Please enable it by configuring a search API key."
)


def search_notes(query: str, index: NotesIndex) -> list[dict]:
    """Return the existing retriever's results without changing their shape."""
    return index.search(query)


def _evaluate(node: ast.AST) -> int | float:
    """Walk only arithmetic nodes; reject names, calls, attributes and objects."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        result = node.value
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        result = _UNARY[type(node.op)](_evaluate(node.operand))
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        # Limit powers BEFORE calculating: even safe syntax can exhaust memory.
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent must be between -100 and 100.")
        result = _BINARY[type(node.op)](left, right)
    else:
        raise ValueError("Only numbers, + - * / ** %, and parentheses are allowed.")
    if type(result) not in (int, float) or abs(result) > 10 ** 100:
        raise ValueError("Result is too large or is not a real number.")
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("Result must be finite.")
    return result


def calculator(expression: str) -> str:
    """Evaluate bounded arithmetic safely and return a result or error string."""
    try:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Enter a mathematical expression.")
        if len(expression) > 300:
            raise ValueError("Expression is too long (maximum 300 characters).")
        tree = ast.parse(expression.strip(), mode="eval")
        # Bounding the tree also limits recursive evaluation depth.
        if sum(1 for _ in ast.walk(tree)) > 100:
            raise ValueError("Expression is too complex.")
        return str(_evaluate(tree.body))
    except (ValueError, SyntaxError, TypeError, ArithmeticError, RecursionError) as exc:
        return f"Error: {exc}"


def web_search(query: str) -> str:
    """Use optional DDGS if installed; missing packages never break imports."""
    try:
        from ddgs import DDGS  # Imported only when the student requests web search.
    except ImportError:
        return _DISABLED
    if not isinstance(query, str) or not query.strip():
        return "Error: Enter a web search query."
    try:
        results = list(DDGS(timeout=10).text(query, max_results=3))
        return "\n\n".join(
            f"{item.get('title', 'Result')}\n{item.get('body', '')}\n"
            f"{item.get('href', '')}" for item in results
        ) or "No web results found."
    except Exception as exc:
        return f"Error: Web search failed: {exc}"
