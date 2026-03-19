"""
Tests for src.systems.long_context

Uses monkeypatching to avoid real API calls.
Tests focus on:
  - Correct prompt construction
  - Token counting and cost calculation
  - LC-Windowed truncation logic
  - QAResult field population
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.systems.base import QAResult
from src.systems.long_context import LongContextSystem, GPT4O_MAX_CONTEXT_CHARS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_cleaned_dir(tmp_path, sample_text):
    """Create a temp cleaned dir with one verdict file."""
    txt = tmp_path / "test_verdict_001.txt"
    txt.write_text(sample_text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def lc_system(tmp_cleaned_dir):
    return LongContextSystem(
        model="gemini-2.5-flash-preview-04-17",
        cleaned_dir=tmp_cleaned_dir,
        windowed=False,
    )


@pytest.fixture
def lc_windowed_system(tmp_cleaned_dir):
    return LongContextSystem(
        model="gpt-4o",
        cleaned_dir=tmp_cleaned_dir,
        windowed=True,
    )


# ── condition_name ─────────────────────────────────────────────────────────────

def test_condition_name_lc(lc_system):
    assert lc_system.condition_name == "lc"


def test_condition_name_windowed(lc_windowed_system):
    assert lc_windowed_system.condition_name == "lc_windowed"


# ── _load_text ─────────────────────────────────────────────────────────────────

def test_load_text_returns_content(lc_system, sample_text):
    text = lc_system._load_text("test_verdict_001")
    assert len(text) > 0
    assert "[1.1]" in text


def test_load_text_missing_file_raises(lc_system):
    with pytest.raises(FileNotFoundError):
        lc_system._load_text("nonexistent_verdict")


def test_windowed_truncation_applied_for_long_doc(tmp_path):
    """Windowed LC should truncate docs that exceed GPT4O_MAX_CONTEXT_CHARS."""
    long_text = "A" * (GPT4O_MAX_CONTEXT_CHARS + 10_000)
    verdict_file = tmp_path / "long_verdict.txt"
    verdict_file.write_text(long_text, encoding="utf-8")

    system = LongContextSystem(model="gpt-4o", cleaned_dir=tmp_path, windowed=True)
    result_text = system._load_text("long_verdict")

    assert len(result_text) < len(long_text)
    assert "DIPOTONG" in result_text


def test_windowed_no_truncation_for_short_doc(tmp_path, sample_text):
    """Short docs should not be truncated even in windowed mode."""
    verdict_file = tmp_path / "short_verdict.txt"
    verdict_file.write_text(sample_text, encoding="utf-8")

    system = LongContextSystem(model="gpt-4o", cleaned_dir=tmp_path, windowed=True)
    result_text = system._load_text("short_verdict")

    assert "DIPOTONG" not in result_text
    assert result_text == sample_text


# ── query (mocked LLM) ────────────────────────────────────────────────────────

def test_query_returns_qa_result(lc_system):
    with patch.object(lc_system, "_call_llm", return_value="Mahkamah menolak permohonan."):
        result = lc_system.query(
            question="Apa amar putusan dalam perkara ini?",
            verdict_id="test_verdict_001",
            question_id="q001",
        )

    assert isinstance(result, QAResult)
    assert result.answer == "Mahkamah menolak permohonan."
    assert result.question_id == "q001"
    assert result.verdict_id == "test_verdict_001"
    assert result.condition == "lc"


def test_query_populates_efficiency_fields(lc_system):
    with patch.object(lc_system, "_call_llm", return_value="Jawaban test."):
        result = lc_system.query(
            question="Siapa pemohon?",
            verdict_id="test_verdict_001",
        )

    assert result.input_tokens > 0
    assert result.cost_usd >= 0.0
    assert result.latency_s > 0.0


def test_query_no_retrieved_chunks_for_lc(lc_system):
    """LC system should never populate retrieved_chunks."""
    with patch.object(lc_system, "_call_llm", return_value="Jawaban."):
        result = lc_system.query("Q?", "test_verdict_001")
    assert result.retrieved_chunks == []


def test_query_result_to_dict_is_serializable(lc_system):
    """QAResult.to_dict() must be JSON-serializable."""
    import json
    with patch.object(lc_system, "_call_llm", return_value="Jawaban."):
        result = lc_system.query("Q?", "test_verdict_001")
    d = result.to_dict()
    json.dumps(d)   # should not raise
