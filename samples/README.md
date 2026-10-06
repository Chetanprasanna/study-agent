# samples/

Place a sample PDF here to enable the real-PDF test in `tests/test_part1.py`.

Any PDF file ending in `.pdf` in this directory will be picked up automatically by:
    `test_real_pdf_if_available` in `tests/test_part1.py`

## How to get a sample PDF

Option A — Download a free textbook chapter:
    https://openstax.org/

Option B — Use any lecture notes PDF you already have.

Option C — Generate a quick one with Python:
```python
# run this once from the repo root:
from reportlab.pdfgen import canvas
c = canvas.Canvas("samples/sample_notes.pdf")
c.drawString(100, 750, "Chapter 1: Introduction to Biology")
c.drawString(100, 700, "The cell is the basic unit of life.")
c.save()
# requires: pip install reportlab
```

The test will skip gracefully if no PDF is present.
