"""
tests/test_table_text_layer_gate.py
-----------------------------------
The OCR-derived-table gate: pipeline/image_processor.py::
table_html_is_reliable()'s page_text_chars check, and the measurement
that feeds it (pipeline/extractor.py::_pdf_page_text_chars()).

2026-09-23 — locks in the fix for a confirmed live wrong answer. The
Godrej Q2 FY26 annexure pages draw their tables as vector outlines, so
those pages carry ~77 characters of real text; unstructured's hi_res path
could only OCR them, and OCR dropped most rows of a 23-row numeric table
while still emitting structurally valid <table> markup. Chat answered
"21.79 million sq ft" for a figure that is actually 34.01, and nothing
logged an error.

Detecting the garbling in the OUTPUT was tried and measured against the
real tables in that file, and rejected: the broken tables score a perfect
1.0 on cell-count uniformity and carry a LOWER junk-character ratio than
several genuinely clean tables in the same document. The source page's
text layer is the signal that actually separates the two cases.
"""

from __future__ import annotations

from unittest.mock import patch

from pipeline.extractor import _pdf_page_text_chars
from pipeline.image_processor import table_html_is_reliable, settings


# A structurally valid table — passes both pre-existing structural checks,
# exactly like the garbled OCR output did.
_WELL_FORMED_HTML = "<table>" + "<tr><td>a</td><td>b</td></tr>" * 5 + "</table>"


class TestStructuralChecksUnchanged:
    """The original two checks must behave exactly as before."""

    def test_empty_html_is_unreliable(self):
        assert table_html_is_reliable("") is False
        assert table_html_is_reliable(None) is False

    def test_too_short_is_unreliable(self):
        assert table_html_is_reliable("<table><tr><td>x</td></tr></table>") is False

    def test_single_row_is_unreliable(self):
        html = "<table><tr><td>" + "x" * 100 + "</td></tr></table>"
        assert table_html_is_reliable(html) is False

    def test_well_formed_table_is_reliable(self):
        assert table_html_is_reliable(_WELL_FORMED_HTML) is True


class TestTextLayerGate:
    def test_page_with_no_text_layer_is_unreliable(self):
        """The actual regression: a structurally perfect table from a
        vector-drawn page (77 chars) is OCR output and must NOT be
        trusted — it goes to the Vision crop path instead."""
        assert table_html_is_reliable(_WELL_FORMED_HTML, page_text_chars=77) is False

    def test_page_with_real_text_layer_stays_reliable(self):
        """Every text-layer table page in the same document measured 477+
        chars — those extract correctly and must keep using text_as_html,
        which is higher fidelity than a Vision prose caption."""
        assert table_html_is_reliable(_WELL_FORMED_HTML, page_text_chars=477) is True

    def test_unknown_page_text_is_treated_as_fine(self):
        """None = non-PDF source, or the measurement failed. Must preserve
        the original behaviour rather than sending every table to Vision."""
        assert table_html_is_reliable(_WELL_FORMED_HTML, page_text_chars=None) is True

    def test_threshold_is_configurable(self):
        with patch.object(settings, "TABLE_TEXT_LAYER_MIN_CHARS", 50):
            assert table_html_is_reliable(_WELL_FORMED_HTML, page_text_chars=77) is True
        with patch.object(settings, "TABLE_TEXT_LAYER_MIN_CHARS", 1000):
            assert table_html_is_reliable(_WELL_FORMED_HTML, page_text_chars=477) is False

    def test_structural_failure_still_wins_over_a_good_text_layer(self):
        """A page with plenty of text but no usable table markup is still
        unreliable — the text-layer check only ever ADDS a reason to
        distrust, it never rescues a broken extraction."""
        assert table_html_is_reliable("", page_text_chars=5000) is False


class TestPdfPageTextChars:
    def test_unreadable_file_returns_empty_dict_never_raises(self):
        """Non-fatal by design: extraction must not fail just because the
        text-layer measurement did. An empty dict makes every page
        'unknown', which table_html_is_reliable() treats as 'assume fine'."""
        assert _pdf_page_text_chars("D:/definitely/not/a/real/file.pdf") == {}

    def test_missing_pymupdf_degrades_gracefully(self):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pymupdf":
                raise ImportError("simulated missing dependency")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            assert _pdf_page_text_chars("anything.pdf") == {}
