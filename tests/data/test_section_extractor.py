"""Tests for src.data.section_extractor"""

import pytest
from src.data.section_extractor import extract_sections


def test_section_extraction_basic(sample_text):
    sections = extract_sections(sample_text, verdict_id="test")
    assert sections.verdict_id == "test"
    assert len(sections.pembukaan) >= 1
    assert len(sections.duduk_perkara) >= 1
    assert len(sections.pertimbangan) >= 1
    assert len(sections.konklusi) >= 1


def test_amar_putusan_extracted(sample_text):
    sections = extract_sections(sample_text, verdict_id="test")
    assert sections.amar_putusan is not None
    assert "Menolak" in sections.amar_putusan


def test_section_text_method(sample_text):
    sections = extract_sections(sample_text, verdict_id="test")
    text = sections.section_text("pertimbangan")
    assert len(text) > 0
    assert "[3." in text


def test_all_sections_text(sample_text):
    sections = extract_sections(sample_text, verdict_id="test")
    full = sections.all_sections_text()
    assert len(full) > 100


def test_all_sections_text_with_exclude(sample_text):
    sections = extract_sections(sample_text, verdict_id="test")
    without_amar = sections.all_sections_text(exclude=["amar_putusan"])
    with_amar = sections.all_sections_text()
    assert len(with_amar) >= len(without_amar)


def test_marker_patterns():
    text = "[1.1] Pembukaan.\n\n[3.5.2] Sub-pertimbangan khusus.\n\n[4.1] Konklusi."
    sections = extract_sections(text)
    assert any(p["marker"] == "[1.1]" for p in sections.pembukaan)
    assert any(p["marker"] == "[4.1]" for p in sections.konklusi)
