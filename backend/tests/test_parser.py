"""Independent tests for the PDF parser (no FastAPI, no LLM)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.parser import PdfParseError, parse_pdf


def build_simple_pdf(page_texts: list[str]) -> bytes:
    """Build a small valid PDF with extractable text on each page."""
    n = len(page_texts)
    page_ids = [4 + (2 * i) for i in range(n)]
    content_ids = [pid + 1 for pid in page_ids]

    objs: dict[int, str] = {}
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    objs[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>"
    objs[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for text, pid, cid in zip(page_texts, page_ids, content_ids):
        stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET"
        objs[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {cid} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objs[cid] = f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream"

    header = b"%PDF-1.4\n"
    body = b""
    offsets = {0: 0}
    pos = len(header)
    max_id = max(objs)
    for i in range(1, max_id + 1):
        chunk = f"{i} 0 obj\n{objs[i]}\nendobj\n".encode("latin-1")
        offsets[i] = pos
        body += chunk
        pos += len(chunk)

    xref_start = pos
    xref_lines = [f"xref\n0 {max_id + 1}\n", "0000000000 65535 f \n"]
    for i in range(1, max_id + 1):
        xref_lines.append(f"{offsets[i]:010d} 00000 n \n")
    trailer = (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    )
    return header + body + "".join(xref_lines).encode("latin-1") + trailer.encode("latin-1")


def _write_temp_pdf(payload: bytes, suffix: str = ".pdf") -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(payload)
    handle.close()
    return Path(handle.name)


class ParsePdfTests(unittest.TestCase):
    def test_extracts_text_and_page_metadata(self) -> None:
        path = _write_temp_pdf(build_simple_pdf(["Hello World"]), suffix="_sample.pdf")
        self.addCleanup(path.unlink)

        pages = parse_pdf(path)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertIn("Hello World", pages[0]["text"])
        self.assertEqual(pages[0]["metadata"]["filename"], path.name)
        self.assertEqual(pages[0]["metadata"]["page_number"], 1)
        self.assertEqual(pages[0]["metadata"]["extracted_text"], pages[0]["text"])

    def test_extracts_each_page_separately(self) -> None:
        path = _write_temp_pdf(build_simple_pdf(["First page", "Second page"]))
        self.addCleanup(path.unlink)

        pages = parse_pdf(path)

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[1]["page_number"], 2)
        self.assertIn("First page", pages[0]["text"])
        self.assertIn("Second page", pages[1]["text"])

    def test_empty_extracted_text_becomes_empty_string(self) -> None:
        path = _write_temp_pdf(build_simple_pdf([""]))
        self.addCleanup(path.unlink)

        pages = parse_pdf(path)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["text"], "")
        self.assertEqual(pages[0]["metadata"]["extracted_text"], "")

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(PdfParseError):
            parse_pdf(Path("definitely-missing.pdf"))

    def test_corrupted_pdf_raises(self) -> None:
        path = _write_temp_pdf(b"%PDF-1.4 this is not a real pdf")
        self.addCleanup(path.unlink)

        with self.assertRaises(PdfParseError):
            parse_pdf(path)


if __name__ == "__main__":
    unittest.main()
