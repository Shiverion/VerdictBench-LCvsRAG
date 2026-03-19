"""Tests for src.data.cleaner"""

import pytest
from src.data.cleaner import clean_text


def test_crlf_normalization():
    text = "line one\r\nline two\r\nline three"
    result = clean_text(text)
    assert "\r" not in result
    assert "line one\nline two" in result


def test_hyphen_break_repair():
    text = "berlak-\nunya undang-undang"
    result = clean_text(text)
    assert "berlakunya" in result


def test_page_number_removal():
    text = "some text\n\n7\n\nmore text"
    result = clean_text(text)
    assert "\n7\n" not in result
    assert "some text" in result
    assert "more text" in result


def test_multiple_blank_lines_collapsed():
    text = "paragraph one\n\n\n\n\nparagraph two"
    result = clean_text(text)
    assert "\n\n\n" not in result
    assert "paragraph one" in result
    assert "paragraph two" in result


def test_empty_text():
    assert clean_text("") == ""


def test_preserves_section_markers():
    text = "[3.1] Menimbang bahwa Mahkamah berwenang.\r\n\r\n[3.2] Menimbang lebih lanjut."
    result = clean_text(text)
    assert "[3.1]" in result
    assert "[3.2]" in result
