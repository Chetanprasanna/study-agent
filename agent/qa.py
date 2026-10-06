"""
agent/qa.py  —  Part 2

Implements the Q&A pipeline:
  1. Find relevant chunks from the index (retrieval)
  2. Build a prompt that includes those chunks as "context"
  3. Ask Gemini to answer the question ONLY from that context
  4. Return the answer + the source chunks used

Why retrieval-augmented generation (RAG)?
  LLMs like Gemini are trained on public internet data — they don't know about
  YOUR notes. RAG solves this by:
    a) Searching the notes for relevant passages (retrieval step)
    b) Injecting those passages into the prompt as context (augmentation step)
    c) Asking the LLM to answer only from that context (generation step)
  This means the model answers from YOUR notes, not from its general knowledge.

CONTRACT:
    answer_question(question: str, index: NotesIndex) -> dict
    Returns: {"answer": str, "sources": list[dict]}

This file is complete (Part 2). Do not modify the public function signature.
"""

# ── Third-party imports ───────────────────────────────────────────────────────
from dotenv import load_dotenv   # Load .env so GEMINI_API_KEY is available

# ── Local imports ─────────────────────────────────────────────────────────────
# We import the two modules we built in Part 2.
from agent.retriever import NotesIndex
from agent.llm import ask_llm

# ── Load environment variables ────────────────────────────────────────────────
# Calling load_dotenv() here is a safety measure — if qa.py is imported before
# llm.py, the key is still loaded. Calling it multiple times is harmless.
load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
# How many chunks to retrieve and pass to the model.
# 4 chunks of ~800 chars each = ~3200 chars of context, well within Gemini's limit.
_DEFAULT_TOP_K: int = 4

# The system instruction tells Gemini what role it plays and what rules to follow.
# A clear system instruction dramatically improves answer quality and faithfulness.
_SYSTEM_INSTRUCTION: str = (
    "You are an expert study assistant that helps students understand their notes. "
    "Your job is to answer questions ONLY using the provided context excerpts from "
    "the student's own notes. "
    "Rules you must follow:\n"
    "1. If the answer is in the context, give a clear, concise answer and cite the "
    "   page number(s) like this: (Page 3) or (Pages 2, 5).\n"
    "2. If the context does not contain enough information to answer the question, "
    "   reply EXACTLY with: \"I couldn't find this in the notes.\"\n"
    "3. Never invent facts or use knowledge from outside the provided context.\n"
    "4. Keep answers focused and beginner-friendly."
)


def answer_question(question: str, index: NotesIndex) -> dict:
    """
    Answer a student's question using retrieved notes and Gemini.

    This is the main RAG (Retrieval-Augmented Generation) pipeline:
      retrieve → augment prompt → generate → return

    Parameters
    ----------
    question : str
        The student's natural-language question.
        Example: "What is the role of the nucleus in a cell?"
    index : NotesIndex
        A built NotesIndex containing all the student's notes as TF-IDF vectors.
        Must have been created with NotesIndex(chunks) before calling this.

    Returns
    -------
    dict
        A dict with two keys, as required by the contract:
            {
                "answer": "The nucleus is the control centre...",
                "sources": [
                    {"id": 2, "page": 3, "text": "...", "score": 0.91},
                    ...
                ]
            }
        "answer"  → Gemini's response as a plain string.
        "sources" → the raw chunk dicts returned by index.search(), each
                    including the "score" field. Useful for showing citations in the UI.

    Examples
    --------
    >>> from agent.retriever import NotesIndex
    >>> index = NotesIndex(chunks)
    >>> result = answer_question("What is ATP?", index)
    >>> "answer" in result and "sources" in result
    True
    """

    # ── Step 1: Retrieve relevant chunks ──────────────────────────────────────
    # Ask the index to find the chunks most relevant to the question.
    # top_k=4 means we get at most 4 chunks. Each chunk is ~800 chars.
    sources: list[dict] = index.search(question, top_k=_DEFAULT_TOP_K)

    # ── Step 2: Handle the case where no relevant content was found ───────────
    # If the index is empty OR the question matches nothing (score=0 everywhere),
    # search() returns []. We short-circuit here to avoid sending an empty prompt.
    if not sources:
        return {
            "answer": "I couldn't find this in the notes.",
            "sources": [],
        }

    # ── Step 3: Build the context block ───────────────────────────────────────
    # We format the retrieved chunks into a readable "Context" section.
    # Each chunk is labelled with its page number so the model can cite it.
    #
    # Example output:
    #   [Excerpt from Page 3, score: 0.87]
    #   The mitochondria is the powerhouse of the cell...
    #
    #   [Excerpt from Page 5, score: 0.62]
    #   ATP is produced during cellular respiration...
    context_parts: list[str] = []
    for chunk in sources:
        # Format: header line + blank line + text + blank line as separator
        header = f"[Excerpt from Page {chunk['page']}, relevance score: {chunk['score']:.2f}]"
        context_parts.append(f"{header}\n{chunk['text']}")

    # Join all excerpts with a clear separator so the model can tell them apart.
    context_block: str = "\n\n---\n\n".join(context_parts)

    # ── Step 4: Build the full prompt ─────────────────────────────────────────
    # We inject the context excerpts and the question into a structured prompt.
    # The template is designed to:
    #   a) Make clear what the context is
    #   b) State the question explicitly
    #   c) Remind the model to cite page numbers
    prompt: str = (
        f"CONTEXT FROM STUDENT'S NOTES:\n"
        f"{'=' * 60}\n"
        f"{context_block}\n"
        f"{'=' * 60}\n\n"
        f"STUDENT'S QUESTION:\n{question}\n\n"
        f"Please answer the question using ONLY the context above. "
        f"Cite page numbers in your answer (e.g., 'According to Page 3...'). "
        f"If the context doesn't answer the question, say exactly: "
        f"\"I couldn't find this in the notes.\""
    )

    # ── Step 5: Call Gemini ────────────────────────────────────────────────────
    # ask_llm() handles the API call, retries, and error handling.
    # We pass the system instruction so Gemini knows its role throughout.
    answer: str = ask_llm(prompt=prompt, system=_SYSTEM_INSTRUCTION)

    # ── Step 6: Return the structured result ──────────────────────────────────
    # The contract requires exactly these two keys: "answer" and "sources".
    return {
        "answer": answer,
        "sources": sources,   # Raw chunk dicts including "score" field
    }
