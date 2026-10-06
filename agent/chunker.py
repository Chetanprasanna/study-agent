"""
agent/chunker.py  —  Part 1

Splits the list of pages produced by pdf_loader.py into smaller, overlapping
text chunks suitable for embedding and retrieval.

Why chunk?
  Large language models have limited context windows, and retrieval works better
  on focused snippets than on entire pages. Chunking divides the text into pieces
  of roughly `chunk_size` characters with an `overlap` so that sentences that
  span a boundary are not lost.

CONTRACT:
    chunk_pages(pages, chunk_size=800, overlap=150) -> list[dict]

This file is complete. Do not modify the public function signature.
"""

# ── Standard library imports ──────────────────────────────────────────────────
import re   # Regular expressions — used for splitting text on sentence boundaries


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """
    Split a list of pages into smaller overlapping text chunks.

    The function tries to respect natural text boundaries in this priority order:
      1. Paragraph breaks  (double newline: \\n\\n)
      2. Sentence endings  (". ", "! ", "? ")
      3. Hard character split (fallback when no boundary is found)

    Parameters
    ----------
    pages : list[dict]
        Output of load_pdf() — each item is {"page": int, "text": str}.
    chunk_size : int, optional
        Target maximum number of characters per chunk. Default 800.
        Chunks may be slightly longer if the nearest boundary is past this limit.
    overlap : int, optional
        Number of characters from the END of one chunk to carry over to the START
        of the next chunk. This preserves context across boundaries. Default 150.

    Returns
    -------
    list[dict]
        A flat list of chunk dicts, globally ordered:
            [{"id": 0, "page": 1, "text": "..."}, {"id": 1, "page": 1, "text": "..."}, ...]
        - "id"   : global 0-based index across ALL pages
        - "page" : the PDF page this chunk came from (1-indexed)
        - "text" : the chunk's text content (stripped)

    Examples
    --------
    >>> pages = [{"page": 1, "text": "Hello world. This is page one."}]
    >>> chunks = chunk_pages(pages, chunk_size=20, overlap=5)
    >>> chunks[0]["page"]
    1
    >>> chunks[0]["id"]
    0
    """

    all_chunks: list[dict] = []   # Final flat list of all chunk dicts
    chunk_id: int = 0             # Global counter that increments across ALL pages

    # Iterate over each page dict produced by load_pdf()
    for page_dict in pages:
        page_num: int = page_dict["page"]   # The 1-indexed page number
        text: str = page_dict["text"]       # The full text of this page

        # Split this page's text into raw chunks (list of strings)
        raw_chunks: list[str] = _split_text(text, chunk_size, overlap)

        # Wrap each raw string in a dict with id and page metadata
        for raw in raw_chunks:
            stripped = raw.strip()

            # Skip any chunk that is empty after stripping (can happen at edges)
            if not stripped:
                continue

            all_chunks.append({
                "id": chunk_id,      # Global unique ID for this chunk
                "page": page_num,    # Which PDF page this chunk came from
                "text": stripped,    # The actual text content
            })
            chunk_id += 1  # Advance the global counter for the next chunk

    return all_chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Internal helper: split a single text string into overlapping chunks.

    Strategy:
    ---------
    We use a sliding-window approach:
      - Start at position 0 in the text.
      - Try to cut at the `chunk_size` position, but look BACK for a nice boundary
        (paragraph break or sentence end) to avoid chopping mid-sentence.
      - The next window STARTS `overlap` characters before the cut point, so
        the previous chunk's tail becomes this chunk's head.

    Parameters
    ----------
    text : str
        The full text of one PDF page.
    chunk_size : int
        Target character count per chunk.
    overlap : int
        Characters of overlap between consecutive chunks.

    Returns
    -------
    list[str]
        Raw chunk strings (may still have leading/trailing whitespace).
    """

    chunks: list[str] = []   # Will hold the resulting chunk strings

    # If the entire text fits in one chunk, no splitting needed → return as-is.
    if len(text) <= chunk_size:
        return [text]

    start: int = 0  # Current window start position in `text`

    # Keep making chunks as long as there is text left to process
    while start < len(text):

        # Determine the tentative end of this chunk
        end: int = start + chunk_size

        # If we've gone past the end of the string, clamp to the string length
        if end >= len(text):
            # This is the last chunk — take everything that remains
            chunks.append(text[start:])
            break  # No more text to process

        # ── Try to find a good split boundary ────────────────────────────────
        # We search BACKWARDS from `end` toward `start` for a natural break.
        # We prefer paragraph breaks first, then sentence endings.

        split_pos: int = _find_split_boundary(text, start, end)

        # Extract the chunk text from `start` to `split_pos`
        chunk_text: str = text[start:split_pos]
        chunks.append(chunk_text)

        # ── Move the window forward, keeping `overlap` characters ─────────────
        # The next chunk starts `overlap` characters BEFORE where we just cut,
        # so the reader (or model) sees the tail of the previous chunk again.
        # This prevents losing context when a sentence straddles a boundary.
        next_start: int = split_pos - overlap

        # Safety: never go backwards (could happen if overlap >= chunk sliced)
        if next_start <= start:
            next_start = split_pos  # Fall back to hard-advancing past the cut

        start = next_start

    return chunks


def _find_split_boundary(text: str, start: int, end: int) -> int:
    """
    Search backwards from `end` toward `start` for a natural split point.

    Priority order:
      1. Paragraph break (\\n\\n) — the cleanest natural boundary
      2. Sentence-ending punctuation followed by a space (".", "!", "?")
      3. Any whitespace character (word boundary fallback)
      4. Hard cut at `end` (last resort)

    Parameters
    ----------
    text : str
        The full text being chunked.
    start : int
        The beginning of the current window (we never split before this).
    end : int
        The target end of the current chunk.

    Returns
    -------
    int
        The character index in `text` where we should cut.
    """

    # ── 1. Look for paragraph break (\\n\\n) ─────────────────────────────────
    # rfind searches from RIGHT to LEFT, so we get the LAST occurrence before `end`.
    para_pos: int = text.rfind("\n\n", start, end)
    if para_pos != -1:
        # Found a paragraph break — cut AFTER the double newline
        # (+2 to move past the "\n\n" itself so it stays with the previous chunk)
        return para_pos + 2

    # ── 2. Look for sentence boundary (". ", "! ", "? ") ─────────────────────
    # We use a regex that matches any sentence-ending punctuation followed by a space.
    # We search within the slice text[start:end] for all matches.
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    # Find all sentence boundary positions in the current window
    sentence_matches = list(sentence_pattern.finditer(text, start, end))

    if sentence_matches:
        # Pick the LAST match so we get the largest possible chunk
        last_match = sentence_matches[-1]
        return last_match.end()  # Cut at the position just after the whitespace

    # ── 3. Look for any whitespace (word boundary) ────────────────────────────
    # If no sentence boundary found, at least avoid cutting mid-word.
    space_pos: int = text.rfind(" ", start, end)
    if space_pos != -1:
        return space_pos + 1  # Cut after the space so it stays with the previous chunk

    # ── 4. Hard cut (last resort) ─────────────────────────────────────────────
    # No boundary found → cut exactly at `end`. This can split a word,
    # but it's rare and better than an infinite loop.
    return end
