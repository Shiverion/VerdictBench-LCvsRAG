"""
Tests for src.systems.simple_rag

Uses an in-memory mock VerdictIndexRegistry to avoid real API calls
and real FAISS index builds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.indexing.chunkers.fixed_size import Chunk
from src.indexing.vector_store import VerdictIndex, VerdictIndexRegistry
from src.systems.base import QAResult
from src.systems.simple_rag import SimpleRAGSystem


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_registry(sample_chunks: list[Chunk]) -> VerdictIndexRegistry:
    """Build a mock registry that returns fixed chunks for any verdict_id."""
    mock_index = MagicMock(spec=VerdictIndex)
    mock_index.search.return_value = sample_chunks[:3]

    mock_registry = MagicMock(spec=VerdictIndexRegistry)
    mock_registry.exists.return_value = True
    mock_registry.get.return_value = mock_index
    return mock_registry


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_cleaned_dir(tmp_path, sample_text):
    f = tmp_path / "test_verdict_001.txt"
    f.write_text(sample_text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def rag_system(tmp_cleaned_dir, sample_chunks):
    registry = _make_mock_registry(sample_chunks)
    return SimpleRAGSystem(
        model="gemini-2.5-flash-preview-04-17",
        chunk_size=512,
        top_k=3,
        registry=registry,
        cleaned_dir=tmp_cleaned_dir,
    )


# ── condition_name ─────────────────────────────────────────────────────────────

def test_condition_name_format(rag_system):
    name = rag_system.condition_name
    assert "simple_rag" in name
    assert "512" in name   # chunk size
    assert "3" in name     # top_k


# ── query ─────────────────────────────────────────────────────────────────────

def test_query_returns_qa_result(rag_system):
    with patch.object(rag_system, "_call_llm", return_value="Mahkamah menolak permohonan."):
        result = rag_system.query(
            question="Apa amar putusan?",
            verdict_id="test_verdict_001",
            question_id="q001",
        )
    assert isinstance(result, QAResult)
    assert result.answer == "Mahkamah menolak permohonan."
    assert result.condition.startswith("simple_rag")


def test_query_populates_retrieved_chunks(rag_system, sample_chunks):
    with patch.object(rag_system, "_call_llm", return_value="Jawaban."):
        result = rag_system.query("Q?", "test_verdict_001")
    assert len(result.retrieved_chunks) > 0
    assert all(isinstance(c, str) for c in result.retrieved_chunks)


def test_query_tracks_input_tokens(rag_system):
    with patch.object(rag_system, "_call_llm", return_value="Jawaban."):
        result = rag_system.query("Q?", "test_verdict_001")
    assert result.input_tokens > 0


def test_query_tracks_cost(rag_system):
    with patch.object(rag_system, "_call_llm", return_value="Jawaban."):
        result = rag_system.query("Q?", "test_verdict_001")
    assert result.cost_usd >= 0.0


def test_query_tracks_latency(rag_system):
    with patch.object(rag_system, "_call_llm", return_value="Jawaban."):
        result = rag_system.query("Q?", "test_verdict_001")
    assert result.latency_s > 0.0


def test_query_records_chunk_size_and_top_k(rag_system):
    with patch.object(rag_system, "_call_llm", return_value="Jawaban."):
        result = rag_system.query("Q?", "test_verdict_001")
    assert result.chunk_size == 512
    assert result.top_k == 3


def test_query_extra_contains_retrieval_latency(rag_system):
    with patch.object(rag_system, "_call_llm", return_value="Jawaban."):
        result = rag_system.query("Q?", "test_verdict_001")
    assert "retrieval_latency_s" in result.extra
    assert result.extra["retrieval_latency_s"] >= 0.0


# ── chunk size variants ────────────────────────────────────────────────────────

@pytest.mark.parametrize("chunk_size,top_k", [
    (256, 3),
    (512, 5),
    (1024, 10),
])
def test_various_chunk_sizes(tmp_cleaned_dir, sample_chunks, chunk_size, top_k):
    registry = _make_mock_registry(sample_chunks)
    system = SimpleRAGSystem(
        model="gemini-2.5-flash-preview-04-17",
        chunk_size=chunk_size,
        top_k=top_k,
        registry=registry,
        cleaned_dir=tmp_cleaned_dir,
    )
    with patch.object(system, "_call_llm", return_value="Jawaban."):
        result = system.query("Q?", "test_verdict_001")
    assert result.chunk_size == chunk_size
    assert result.top_k == top_k


# ── index build on demand ─────────────────────────────────────────────────────

def test_index_built_when_not_exists(tmp_cleaned_dir, sample_chunks):
    """If index doesn't exist, _ensure_index should build it."""
    mock_registry = MagicMock(spec=VerdictIndexRegistry)
    mock_registry.exists.return_value = False  # not yet built
    mock_registry.build_and_save.return_value = MagicMock()
    mock_index = MagicMock(spec=VerdictIndex)
    mock_index.search.return_value = sample_chunks[:2]
    mock_registry.get.return_value = mock_index

    system = SimpleRAGSystem(
        model="gemini-2.5-flash-preview-04-17",
        registry=mock_registry,
        cleaned_dir=tmp_cleaned_dir,
    )

    with patch.object(system, "_call_llm", return_value="Jawaban."):
        system.query("Q?", "test_verdict_001")

    mock_registry.build_and_save.assert_called_once()
