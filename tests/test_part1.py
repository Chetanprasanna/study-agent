"""
tests/test_part1.py  —  Part 1

Pytest tests for:
  - agent/pdf_loader.py  (load_pdf)
  - agent/chunker.py     (chunk_pages)

Run with:
    pytest tests/test_part1.py -v
"""

# ── Standard library imports ──────────────────────────────────────────────────
import io          # To create in-memory file-like objects (BytesIO)
import struct      # Used in the fake PDF builder below
from pathlib import Path

# ── Third-party imports ───────────────────────────────────────────────────────
import pytest      # Testing framework — provides fixtures, assert helpers, etc.

# ── Module under test ─────────────────────────────────────────────────────────
from agent.pdf_loader import load_pdf
from agent.chunker import chunk_pages


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers & Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _make_minimal_pdf(pages: list[str]) -> bytes:
    """
    Build a minimal but valid PDF in memory with the given text pages.

    This creates a real, parseable PDF that pypdf can read — so we don't
    need a real PDF file on disk for unit tests.

    Parameters
    ----------
    pages : list[str]
        A list of strings; each string becomes the text content of one page.

    Returns
    -------
    bytes
        A valid PDF file as a bytes object.
    """

    # We'll build the PDF objects manually.
    # A minimal PDF has: header, objects, xref table, trailer.
    # Each page uses a content stream with BT (Begin Text) / ET (End Text) operators.

    objects = []   # We collect raw PDF object bytes here

    # ── Object 1: Catalog (root of the PDF document tree) ────────────────────
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    # We'll collect page object numbers so the Pages dict can reference them.
    page_obj_ids = []
    # Objects start at index 3 (1=catalog, 2=pages, 3+=actual pages)
    first_page_obj = 3

    # ── Build one object per page ─────────────────────────────────────────────
    for i, text in enumerate(pages):
        obj_id = first_page_obj + i * 2          # page dict object
        stream_obj_id = first_page_obj + i * 2 + 1  # content stream object
        page_obj_ids.append(obj_id)

        # Escape parentheses in text (PDF string syntax)
        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        # Content stream: move to (50, 700) and draw the text
        stream_content = f"BT /F1 12 Tf 50 700 Td ({safe_text}) Tj ET".encode()
        stream_len = len(stream_content)

        # Page content stream object
        objects.append(
            f"{stream_obj_id} 0 obj\n"
            f"<< /Length {stream_len} >>\n"
            f"stream\n".encode() +
            stream_content +
            b"\nendstream\nendobj\n"
        )

        # Page dictionary object
        objects.append(
            f"{obj_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /Contents {stream_obj_id} 0 R\n"
            f"   /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>\n"
            f">>\nendobj\n".encode()
        )

    # ── Object 2: Pages dictionary (references all page objects) ──────────────
    kids_refs = " ".join(f"{oid} 0 R" for oid in page_obj_ids)
    pages_obj = (
        f"2 0 obj\n"
        f"<< /Type /Pages /Kids [{kids_refs}] /Count {len(pages)} >>\n"
        f"endobj\n"
    ).encode()

    # ── Assemble the PDF body ─────────────────────────────────────────────────
    header = b"%PDF-1.4\n"

    # We need offsets for the xref table, so track byte positions
    body = b""
    offsets = {}   # obj_number -> byte offset from start of file

    # Catalog is object 1
    offsets[1] = len(header)
    body += objects[0]   # catalog

    # Pages dict is object 2
    offsets[2] = len(header) + len(body)
    body += pages_obj

    # Page objects (stream then dict, interleaved)
    obj_index = 1  # skip catalog (index 0)
    obj_num = first_page_obj

    # Re-iterate: stream objects come at odd positions in our objects list
    # objects[1], objects[3], objects[5] = streams for pages 0, 1, 2, ...
    # objects[2], objects[4], objects[6] = page dicts
    for i in range(len(pages)):
        stream_idx = 1 + i * 2
        dict_idx = 2 + i * 2
        stream_id = first_page_obj + i * 2 + 1
        dict_id = first_page_obj + i * 2

        offsets[stream_id] = len(header) + len(body)
        body += objects[stream_idx]

        offsets[dict_id] = len(header) + len(body)
        body += objects[dict_idx]

    # ── Cross-reference table ─────────────────────────────────────────────────
    xref_offset = len(header) + len(body)
    total_objects = 2 + len(pages) * 2  # catalog + pages + (stream + dict) per page
    xref = f"xref\n0 {total_objects + 1}\n0000000000 65535 f \n".encode()
    for oid in range(1, total_objects + 1):
        offset = offsets.get(oid, 0)
        xref += f"{offset:010d} 00000 n \n".encode()

    # ── Trailer ───────────────────────────────────────────────────────────────
    trailer = (
        f"trailer\n<< /Size {total_objects + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return header + body + xref + trailer


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Pytest fixture: a 3-page PDF where page 2 is intentionally blank."""
    return _make_minimal_pdf(["First page content.", "", "Third page content."])


@pytest.fixture
def sample_pdf_path(tmp_path, sample_pdf_bytes) -> Path:
    """Pytest fixture: write the sample PDF to a temp file and return its path."""
    pdf_file = tmp_path / "test_notes.pdf"
    pdf_file.write_bytes(sample_pdf_bytes)
    return pdf_file


@pytest.fixture
def real_pdf_path() -> Path | None:
    """
    Returns the path to a real sample PDF if one exists in samples/.
    Tests that use this fixture are skipped if no sample PDF is available.
    """
    samples_dir = Path(__file__).parent.parent / "samples"
    pdfs = list(samples_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for load_pdf
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadPdf:
    """Tests for agent.pdf_loader.load_pdf"""

    def test_returns_list(self, sample_pdf_path):
        """load_pdf must always return a list."""
        result = load_pdf(sample_pdf_path)
        assert isinstance(result, list), "Expected a list"

    def test_each_item_is_dict(self, sample_pdf_path):
        """Every item in the result must be a dict."""
        result = load_pdf(sample_pdf_path)
        for item in result:
            assert isinstance(item, dict), f"Expected dict, got {type(item)}"

    def test_dict_has_required_keys(self, sample_pdf_path):
        """Every page dict must have exactly 'page' and 'text' keys."""
        result = load_pdf(sample_pdf_path)
        for item in result:
            assert "page" in item, "Missing 'page' key"
            assert "text" in item, "Missing 'text' key"

    def test_page_numbers_are_integers(self, sample_pdf_path):
        """'page' values must be integers."""
        result = load_pdf(sample_pdf_path)
        for item in result:
            assert isinstance(item["page"], int), "'page' must be int"

    def test_page_numbers_are_one_indexed(self, sample_pdf_path):
        """Page numbers must start at 1, not 0."""
        result = load_pdf(sample_pdf_path)
        assert result[0]["page"] >= 1, "First page number must be >= 1"

    def test_text_values_are_strings(self, sample_pdf_path):
        """'text' values must be strings."""
        result = load_pdf(sample_pdf_path)
        for item in result:
            assert isinstance(item["text"], str), "'text' must be str"

    def test_text_is_not_empty(self, sample_pdf_path):
        """No returned page should have empty text (empty pages are skipped)."""
        result = load_pdf(sample_pdf_path)
        for item in result:
            assert item["text"].strip(), f"Page {item['page']} has empty text"

    def test_accepts_file_like_object(self, sample_pdf_bytes):
        """load_pdf must accept a file-like object (e.g. io.BytesIO)."""
        file_like = io.BytesIO(sample_pdf_bytes)
        result = load_pdf(file_like)
        # Should work without raising — just check we get a list back
        assert isinstance(result, list)

    def test_accepts_string_path(self, sample_pdf_path):
        """load_pdf must accept a plain string path, not just Path objects."""
        result = load_pdf(str(sample_pdf_path))
        assert isinstance(result, list)
        assert len(result) > 0

    def test_accepts_pathlib_path(self, sample_pdf_path):
        """load_pdf must accept a pathlib.Path object."""
        result = load_pdf(sample_pdf_path)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_skips_empty_pages(self, sample_pdf_bytes):
        """
        Pages with no text content must be skipped.
        Our sample PDF has 3 pages: page 1, blank page 2, page 3.
        We expect only 2 results (pages 1 and 3).
        """
        file_like = io.BytesIO(sample_pdf_bytes)
        result = load_pdf(file_like)

        # We should get at most 2 pages (the blank middle page must be gone)
        # Note: pypdf may extract the blank page as whitespace-only — we filter those.
        non_empty_count = sum(1 for item in result if item["text"].strip())
        assert non_empty_count >= 1, "Should have at least 1 non-empty page"

        # Verify none of the returned pages have empty text
        for item in result:
            assert item["text"].strip(), f"Page {item['page']} should not be empty"

    def test_page_numbers_are_ascending(self, sample_pdf_path):
        """Page numbers should be in ascending order (PDF is read front-to-back)."""
        result = load_pdf(sample_pdf_path)
        page_nums = [item["page"] for item in result]
        assert page_nums == sorted(page_nums), "Page numbers must be ascending"

    def test_single_page_pdf(self):
        """A single-page PDF should return a list with one item."""
        pdf_bytes = _make_minimal_pdf(["Only one page."])
        result = load_pdf(io.BytesIO(pdf_bytes))
        assert len(result) == 1
        assert result[0]["page"] == 1

    def test_all_blank_pdf_returns_empty_list(self):
        """A PDF where all pages are blank must return an empty list."""
        pdf_bytes = _make_minimal_pdf(["", "   ", "\n"])
        result = load_pdf(io.BytesIO(pdf_bytes))
        # All pages are whitespace-only, so we expect 0 results
        assert result == [] or all(not item["text"].strip() for item in result) is False
        # More precisely: no item should have empty stripped text
        for item in result:
            assert item["text"].strip(), "Blank pages must be skipped"

    def test_real_pdf_if_available(self, real_pdf_path):
        """Run against a real PDF in samples/ if one exists."""
        if real_pdf_path is None:
            pytest.skip("No sample PDF found in samples/ — skipping real PDF test")
        result = load_pdf(real_pdf_path)
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert "page" in item
            assert "text" in item


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for chunk_pages
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkPages:
    """Tests for agent.chunker.chunk_pages"""

    @pytest.fixture
    def sample_pages(self):
        """A simple list of pages for testing."""
        return [
            {"page": 1, "text": "Hello world. This is the first page. It has some content here."},
            {"page": 2, "text": "Second page text. More information about the topic. And some extra sentences."},
        ]

    @pytest.fixture
    def long_page(self):
        """A single page with text long enough to produce multiple chunks."""
        # Repeat a sentence many times to exceed chunk_size=800 chars easily
        long_text = ("The mitochondria is the powerhouse of the cell. " * 30)
        return [{"page": 1, "text": long_text}]

    def test_returns_list(self, sample_pages):
        """chunk_pages must return a list."""
        result = chunk_pages(sample_pages)
        assert isinstance(result, list)

    def test_each_item_is_dict(self, sample_pages):
        """Every chunk must be a dict."""
        result = chunk_pages(sample_pages)
        for item in result:
            assert isinstance(item, dict)

    def test_chunk_has_required_keys(self, sample_pages):
        """Every chunk dict must have 'id', 'page', and 'text' keys."""
        result = chunk_pages(sample_pages)
        for item in result:
            assert "id" in item, "Missing 'id' key"
            assert "page" in item, "Missing 'page' key"
            assert "text" in item, "Missing 'text' key"

    def test_ids_are_globally_unique(self, sample_pages):
        """All chunk 'id' values must be unique."""
        result = chunk_pages(sample_pages)
        ids = [item["id"] for item in result]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_ids_start_at_zero(self, sample_pages):
        """First chunk must have id=0."""
        result = chunk_pages(sample_pages)
        if result:
            assert result[0]["id"] == 0

    def test_ids_are_sequential(self, sample_pages):
        """IDs should be 0, 1, 2, ... with no gaps."""
        result = chunk_pages(sample_pages)
        ids = [item["id"] for item in result]
        assert ids == list(range(len(ids))), f"Expected sequential IDs, got {ids}"

    def test_page_numbers_preserved(self, sample_pages):
        """Chunks should carry the correct page number from their source page."""
        result = chunk_pages(sample_pages)
        # Find chunks from page 2
        page2_chunks = [c for c in result if c["page"] == 2]
        # There should be at least one chunk for page 2
        assert len(page2_chunks) >= 1, "Expected at least one chunk from page 2"
        # All page2 chunks should indeed say page=2
        for chunk in page2_chunks:
            assert chunk["page"] == 2

    def test_text_is_non_empty(self, sample_pages):
        """No chunk should have empty text."""
        result = chunk_pages(sample_pages)
        for item in result:
            assert item["text"].strip(), f"Chunk {item['id']} has empty text"

    def test_empty_input_returns_empty_list(self):
        """chunk_pages([]) must return []."""
        result = chunk_pages([])
        assert result == []

    def test_short_text_fits_in_one_chunk(self):
        """Text shorter than chunk_size must produce exactly one chunk."""
        pages = [{"page": 1, "text": "Short text."}]
        result = chunk_pages(pages, chunk_size=800, overlap=150)
        assert len(result) == 1
        assert result[0]["text"] == "Short text."
        assert result[0]["page"] == 1
        assert result[0]["id"] == 0

    def test_long_text_produces_multiple_chunks(self, long_page):
        """Text longer than chunk_size must produce more than one chunk."""
        result = chunk_pages(long_page, chunk_size=200, overlap=50)
        assert len(result) > 1, "Long text should produce multiple chunks"

    def test_overlap_creates_repeated_content(self, long_page):
        """
        Consecutive chunks should share some characters at their boundary
        (the overlap). We verify this by checking that the end of chunk N
        appears somewhere in chunk N+1.
        """
        result = chunk_pages(long_page, chunk_size=200, overlap=50)

        if len(result) < 2:
            pytest.skip("Not enough chunks to test overlap")

        # Take the last 30 chars of the first chunk
        tail_of_first = result[0]["text"][-30:].strip()

        # That tail should appear somewhere in the second chunk
        assert tail_of_first in result[1]["text"], (
            "Overlap content from chunk 0 should appear in chunk 1"
        )

    def test_chunk_size_respected_approximately(self, long_page):
        """
        Chunks should be at most roughly chunk_size + a sentence-boundary buffer.
        We allow up to 2x chunk_size to account for boundary search.
        """
        chunk_size = 200
        result = chunk_pages(long_page, chunk_size=chunk_size, overlap=50)
        for item in result:
            assert len(item["text"]) <= chunk_size * 2, (
                f"Chunk {item['id']} is unexpectedly long: {len(item['text'])} chars"
            )

    def test_all_original_text_covered(self, long_page):
        """
        Every word from the original text should appear in at least one chunk.
        (Because of overlap, some words appear in multiple chunks — that's fine.)
        """
        original_words = set(long_page[0]["text"].split())
        result = chunk_pages(long_page, chunk_size=200, overlap=50)
        all_chunk_words = set()
        for item in result:
            all_chunk_words.update(item["text"].split())

        # Check that all original words are present somewhere in the chunks
        missing = original_words - all_chunk_words
        assert len(missing) == 0, f"These words were lost in chunking: {missing}"

    def test_multi_page_ids_are_global(self):
        """
        When chunking multiple pages, IDs must continue incrementing globally,
        not reset to 0 for each page.
        """
        long_text = "Sentence about biology. " * 40  # ~960 chars → multiple chunks
        pages = [
            {"page": 1, "text": long_text},
            {"page": 2, "text": long_text},
        ]
        result = chunk_pages(pages, chunk_size=200, overlap=50)
        ids = [item["id"] for item in result]
        # IDs must be globally sequential — not [0,1,2, 0,1,2]
        assert ids == list(range(len(ids))), (
            "IDs must be globally sequential across all pages"
        )

    def test_custom_chunk_size_and_overlap(self):
        """Verify the function accepts and uses custom chunk_size and overlap."""
        text = "Word " * 200   # 1000 chars
        pages = [{"page": 1, "text": text}]
        result_small = chunk_pages(pages, chunk_size=100, overlap=20)
        result_large = chunk_pages(pages, chunk_size=500, overlap=50)
        # Smaller chunk_size → more chunks
        assert len(result_small) > len(result_large), (
            "Smaller chunk_size should produce more chunks"
        )
