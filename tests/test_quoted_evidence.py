"""
tests/test_quoted_evidence.py
-------------------------------
The "exact quote" citation feature added 2026-09-23:
pipeline/memory.py's QUOTED EVIDENCE system-prompt rule asks the LLM to
append a "===QUOTES===" block after its answer — one line per cited
source, with a verbatim quote — in the SAME completion (no extra LLM
call). api/routes/chat.py parses that block out, verifies each quote is
a real substring of its source chunk before trusting it, and attaches it
to that source's entry as `quoted_text`. Also covers the top-N sources
cut (settings.MAX_SOURCES_RETURNED) that shipped alongside it.
"""

from __future__ import annotations

from api.routes.chat import (
    _split_answer_and_quotes, _verified_quote_for_chunk, _build_sources, _normalize_for_match,
)
from pipeline.memory import source_tag, QUOTES_SENTINEL


class TestSplitAnswerAndQuotes:
    def test_no_sentinel_returns_content_unchanged_and_no_quotes(self):
        answer, quotes = _split_answer_and_quotes("Revenue was $5M. [Source: report.pdf, page 2]")
        assert answer == "Revenue was $5M. [Source: report.pdf, page 2]"
        assert quotes == {}

    def test_splits_clean_answer_from_quotes_block(self):
        raw = (
            "Revenue was $5M. [Source: report.pdf, page 2]\n"
            f"{QUOTES_SENTINEL}\n"
            '[Source: report.pdf, page 2] "Total revenue for the quarter was $5.0M."\n'
        )
        answer, quotes = _split_answer_and_quotes(raw)
        assert answer == "Revenue was $5M. [Source: report.pdf, page 2]"
        assert quotes == {"[Source: report.pdf, page 2]": "Total revenue for the quarter was $5.0M."}

    def test_multiple_quote_lines_all_parsed(self):
        raw = (
            "Combined this quarter. [Source: a.pdf, page 1] [Source: b.pdf, page 3]\n"
            f"{QUOTES_SENTINEL}\n"
            '[Source: a.pdf, page 1] "First figure was 10."\n'
            '[Source: b.pdf, page 3] "Second figure was 20."\n'
        )
        _, quotes = _split_answer_and_quotes(raw)
        assert quotes == {
            "[Source: a.pdf, page 1]": "First figure was 10.",
            "[Source: b.pdf, page 3]": "Second figure was 20.",
        }

    def test_malformed_quote_line_is_skipped_not_crashed_on(self):
        raw = f"Answer text.\n{QUOTES_SENTINEL}\nthis line has no tag or quotes\n"
        answer, quotes = _split_answer_and_quotes(raw)
        assert answer == "Answer text."
        assert quotes == {}


class TestVerifiedQuoteForChunk:
    def test_verbatim_quote_in_chunk_text_is_kept(self):
        chunk = {"doc_name": "report.pdf", "page_number": 2, "chunk_text": "Total revenue for the quarter was $5.0M, up 10% YoY."}
        tag = source_tag(chunk)
        result = _verified_quote_for_chunk(chunk, {tag: "Total revenue for the quarter was $5.0M, up 10% YoY."})
        assert result == "Total revenue for the quarter was $5.0M, up 10% YoY."

    def test_quote_not_present_in_chunk_is_dropped(self):
        """The exact failure mode this guards against: the model reconstructs
        a quote from memory instead of copying it -- must not be trusted."""
        chunk = {"doc_name": "report.pdf", "page_number": 2, "chunk_text": "Total revenue was $5.0M."}
        tag = source_tag(chunk)
        result = _verified_quote_for_chunk(chunk, {tag: "Total revenue was six million dollars."})
        assert result is None

    def test_no_matching_tag_returns_none(self):
        chunk = {"doc_name": "report.pdf", "page_number": 2, "chunk_text": "Some text."}
        result = _verified_quote_for_chunk(chunk, {"[Source: other.pdf, page 9]": "unrelated"})
        assert result is None

    def test_whitespace_differences_are_tolerated(self):
        """A quote reproduced with different line-wrapping than the source
        is still a real quote -- only WORD content must match exactly."""
        chunk = {"doc_name": "report.pdf", "page_number": 2, "chunk_text": "Total revenue\nfor the quarter   was $5.0M."}
        tag = source_tag(chunk)
        result = _verified_quote_for_chunk(chunk, {tag: "Total revenue for the quarter was $5.0M."})
        assert result == "Total revenue for the quarter was $5.0M."

    def test_image_chunk_verified_against_caption_not_chunk_text(self):
        chunk = {
            "doc_name": "deck.pdf", "page_number": 5, "role": "image",
            "image_caption": "Bengaluru: 35.9 million sq ft, 18 projects",
            "chunk_text": "[IMAGE] fallback placeholder text",
        }
        tag = source_tag(chunk)
        result = _verified_quote_for_chunk(chunk, {tag: "Bengaluru: 35.9 million sq ft, 18 projects"})
        assert result == "Bengaluru: 35.9 million sq ft, 18 projects"


class TestBuildSourcesTopN:
    def test_only_first_n_chunks_become_sources(self):
        chunks = [
            {"doc_name": f"doc{i}.pdf", "chunk_index": i, "chunk_text": f"text {i}", "page_number": i}
            for i in range(5)
        ]
        sources = _build_sources(chunks[:3], {})
        assert len(sources) == 3
        assert [s["doc_name"] for s in sources] == ["doc0.pdf", "doc1.pdf", "doc2.pdf"]

    def test_quoted_text_attached_when_present_and_verified(self):
        chunk = {"doc_name": "report.pdf", "chunk_index": 0, "chunk_text": "The total was 42 units.", "page_number": 1}
        tag = source_tag(chunk)
        sources = _build_sources([chunk], {tag: "The total was 42 units."})
        assert sources[0]["quoted_text"] == "The total was 42 units."

    def test_quoted_text_none_when_no_quote_provided(self):
        chunk = {"doc_name": "report.pdf", "chunk_index": 0, "chunk_text": "Some content.", "page_number": 1}
        sources = _build_sources([chunk], {})
        assert sources[0]["quoted_text"] is None


class TestNormalizeForMatch:
    def test_collapses_whitespace_runs(self):
        assert _normalize_for_match("a   b\n\nc") == "a b c"

    def test_none_safe(self):
        assert _normalize_for_match(None) == ""
