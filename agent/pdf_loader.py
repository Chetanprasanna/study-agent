"""
agent/pdf_loader.py  —  Part 1

Responsible for reading a PDF file and extracting the text from each page.
Returns a clean list of dicts: [{"page": int, "text": str}, ...]

CONTRACT:
    load_pdf(file) -> list[dict]

This file is complete. Do not modify the public function signature.
"""

# ── Standard library imports ──────────────────────────────────────────────────
import io        # Needed to detect if `file` is already a bytes/file-like object
from pathlib import Path  # Lets us handle file paths in a cross-platform way

# ── Third-party imports ───────────────────────────────────────────────────────
# pypdf is a pure-Python PDF reading library.
# We only import PdfReader — the class that opens and parses the PDF.
from pypdf import PdfReader


def load_pdf(file) -> list[dict]:
    """
    Extract text from every non-empty page of a PDF.

    Parameters
    ----------
    file : str | Path | file-like object
        Either:
          - A file-system path as a string (e.g. "notes/lecture1.pdf")
          - A pathlib.Path object (e.g. Path("notes/lecture1.pdf"))
          - A file-like object with a .read() method (e.g. Streamlit's UploadedFile,
            or any io.BytesIO / open(..., "rb") handle)

    Returns
    -------
    list[dict]
        A list of dicts, one per NON-EMPTY page, in reading order:
            [{"page": 1, "text": "..."}, {"page": 3, "text": "..."}, ...]
        - "page"  : 1-indexed page number (same as the real PDF page number)
        - "text"  : the extracted text for that page (stripped of leading/trailing whitespace)
        Blank pages (pages where all extracted text is whitespace) are SKIPPED.

    Examples
    --------
    >>> pages = load_pdf("my_notes.pdf")
    >>> pages[0]
    {'page': 1, 'text': 'Introduction to Biology...'}

    >>> from io import BytesIO
    >>> with open("my_notes.pdf", "rb") as f:
    ...     data = BytesIO(f.read())
    >>> pages = load_pdf(data)
    """

    # ── Step 1: Open the PDF with PdfReader ──────────────────────────────────
    # PdfReader can accept both file paths and file-like objects directly,
    # but we add an explicit check so error messages are clear.

    if isinstance(file, (str, Path)):
        # It's a path on disk → open it for reading in binary mode ("rb")
        # 'rb' means: Read Binary — PDFs are binary files, not plain text.
        with open(file, "rb") as f:
            reader = PdfReader(f)
            # We extract pages INSIDE the `with` block so the file stays open.
            pages = _extract_pages(reader)
    else:
        # It's already a file-like object (e.g. Streamlit UploadedFile, io.BytesIO).
        # PdfReader can read from it directly without us opening anything.
        reader = PdfReader(file)
        pages = _extract_pages(reader)

    return pages


def _extract_pages(reader: PdfReader) -> list[dict]:
    """
    Internal helper: iterate over all pages in the reader and collect non-empty ones.

    Parameters
    ----------
    reader : PdfReader
        An already-opened PdfReader instance.

    Returns
    -------
    list[dict]
        List of {"page": int, "text": str} dicts for pages that have content.
    """

    results = []  # This list will hold our final output dicts.

    # reader.pages is a list-like object containing one PageObject per PDF page.
    # enumerate gives us both an index (0-based) and the page object.
    for page_index, page in enumerate(reader.pages):

        # ── Extract raw text from this page ──────────────────────────────────
        # .extract_text() is pypdf's built-in method. It returns a string, or
        # None if extraction failed entirely (rare but possible for scanned PDFs).
        raw_text = page.extract_text()

        # Guard against None (pypdf returns None for some scanned pages)
        if raw_text is None:
            continue  # Skip this page entirely

        # .strip() removes leading/trailing whitespace and newlines.
        # We use this cleaned version to check if the page is empty.
        cleaned_text = raw_text.strip()

        # ── Skip blank pages ─────────────────────────────────────────────────
        # An empty page (or a page with only whitespace) has no useful content,
        # so we skip it to avoid polluting our chunk index with useless entries.
        if not cleaned_text:
            continue  # move on to the next page

        # ── Build the page dict and append ───────────────────────────────────
        # Page numbers in PDFs are 0-indexed internally but humans expect 1-indexed.
        # So we add 1 to page_index.
        results.append({
            "page": page_index + 1,   # 1-indexed page number
            "text": cleaned_text,     # cleaned, non-empty text
        })

    return results
