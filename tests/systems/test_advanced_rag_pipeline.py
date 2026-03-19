"""
Tests for src.systems.advanced_rag.pipeline

Covers:
  - AblationFlags factory methods and condition_suffix
  - AdvancedRAGSystem condition_name composition
  - Component activation/deactivation via flags
  - QAResult field correctness
  - query_rewriter, reranker, metadata_filter unit behaviour
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.indexing.chunkers.fixed_size import Chunk
from src.indexing.vector_store import VerdictIndex, VerdictIndexRegistry
from src.systems.advanced_rag.metadata_filter import (
    MetadataFilter,
    apply_filter,
    infer_filter_from_question,
)
from src.systems.advanced_rag.pipeline import AdvancedRAGSystem, AblationFlags
from src.systems.base import QAResult


# ── AblationFlags ─────────────────────────────────────────────────────────────

def test_simple_rag_flags_all_false():
    flags = AblationFlags.simple_rag()
    assert not flags.use_query_rewrite
    assert not flags.use_metadata_filter
    assert not flags.use_hybrid_search
    assert not flags.use_reranking


def test_full_advanced_flags_all_true():
    flags = AblationFlags.full_advanced()
    assert flags.use_query_rewrite
    assert flags.use_metadata_filter
    assert flags.use_hybrid_search
    assert flags.use_reranking


def test_query_rewrite_only():
    flags = AblationFlags.query_rewrite_only()
    assert flags.use_query_rewrite
    assert not flags.use_reranking
    assert not flags.use_hybrid_search
    assert not flags.use_metadata_filter


def test_condition_suffix_simple():
    assert AblationFlags.simple_rag().condition_suffix == "simple_rag"


def test_condition_suffix_full():
    suffix = AblationFlags.full_advanced().condition_suffix
    assert "QR" in suffix
    assert "MF" in suffix
    assert "HS" in suffix
    assert "RR" in suffix


def test_condition_suffix_single():
    assert AblationFlags.reranking_only().condition_suffix == "RR"
    assert AblationFlags.hybrid_only().condition_suffix == "HS"


# ── AdvancedRAGSystem.condition_name ──────────────────────────────────────────

def test_condition_name_includes_flags():
    system = AdvancedRAGSystem(
        model="gemini-2.5-flash-preview-04-17",
        flags=AblationFlags.full_advanced(),
    )
    assert system.condition_name.startswith("advanced_rag_")
    assert "QR" in system.condition_name


# ── AdvancedRAGSystem.query (mocked) ─────────────────────────────────────────

def _make_rag_system(tmp_cleaned_dir, sample_chunks, flags):
    mock_index = MagicMock(spec=VerdictIndex)
    mock_index.search.return_value = sample_chunks

    mock_registry = MagicMock(spec=VerdictIndexRegistry)
    mock_registry.exists.return_value = True
    mock_registry.get.return_value = mock_index

    return AdvancedRAGSystem(
        model="gemini-2.5-flash-preview-04-17",
        flags=flags,
        registry=mock_registry,
        cleaned_dir=tmp_cleaned_dir,
    )


@pytest.fixture
def tmp_cleaned_dir(tmp_path, sample_text):
    f = tmp_path / "test_verdict_001.txt"
    f.write_text(sample_text, encoding="utf-8")
    return tmp_path


def test_query_simple_rag_baseline(tmp_cleaned_dir, sample_chunks):
    system = _make_rag_system(tmp_cleaned_dir, sample_chunks, AblationFlags.simple_rag())
    with patch.object(system, "_call_llm", return_value="Jawaban baseline."):
        result = system.query("Q?", "test_verdict_001", "q001")
    assert isinstance(result, QAResult)
    assert result.answer == "Jawaban baseline."
    assert "simple_rag" in result.condition


def test_query_full_advanced_calls_rewrite(tmp_cleaned_dir, sample_chunks):
    system = _make_rag_system(tmp_cleaned_dir, sample_chunks, AblationFlags.full_advanced())
    with patch("src.systems.advanced_rag.pipeline.rewrite_query",
               return_value=["rewritten query"]) as mock_rw, \
         patch.object(system, "_call_llm", return_value="Jawaban."), \
         patch("src.systems.advanced_rag.pipeline.rerank",
               return_value=sample_chunks[:2]):
        result = system.query("Q?", "test_verdict_001")
    mock_rw.assert_called_once()
    assert "query_rewrite" in result.extra.get("components_used", [])


def test_query_reranking_only_calls_rerank(tmp_cleaned_dir, sample_chunks):
    system = _make_rag_system(tmp_cleaned_dir, sample_chunks, AblationFlags.reranking_only())
    with patch("src.systems.advanced_rag.pipeline.rerank",
               return_value=sample_chunks[:1]) as mock_rr, \
         patch.object(system, "_call_llm", return_value="Jawaban."), \
         patch("src.systems.advanced_rag.pipeline.rewrite_query",
               return_value=["Q?"]):
        result = system.query("Q?", "test_verdict_001")
    mock_rr.assert_called_once()
    assert "reranking" in result.extra.get("components_used", [])


def test_query_no_rewrite_when_flag_false(tmp_cleaned_dir, sample_chunks):
    system = _make_rag_system(tmp_cleaned_dir, sample_chunks, AblationFlags.simple_rag())
    with patch("src.systems.advanced_rag.pipeline.rewrite_query") as mock_rw, \
         patch.object(system, "_call_llm", return_value="Jawaban."):
        system.query("Q?", "test_verdict_001")
    mock_rw.assert_not_called()


def test_query_extra_contains_component_list(tmp_cleaned_dir, sample_chunks):
    system = _make_rag_system(tmp_cleaned_dir, sample_chunks, AblationFlags.simple_rag())
    with patch.object(system, "_call_llm", return_value="Jawaban."):
        result = system.query("Q?", "test_verdict_001")
    assert "components_used" in result.extra
    assert isinstance(result.extra["components_used"], list)


# ── metadata_filter unit tests ────────────────────────────────────────────────

def test_apply_filter_section_whitelist(sample_chunks):
    f = MetadataFilter(sections=["pertimbangan"])
    filtered = apply_filter(sample_chunks, f)
    assert all(c.section == "pertimbangan" for c in filtered)


def test_apply_filter_section_blacklist(sample_chunks):
    f = MetadataFilter(exclude_sections=["amar_putusan"])
    filtered = apply_filter(sample_chunks, f)
    assert all(c.section != "amar_putusan" for c in filtered)


def test_apply_filter_min_chars(sample_chunks):
    f = MetadataFilter(min_chars=1000)   # all sample chunks are shorter
    filtered = apply_filter(sample_chunks, f)
    assert len(filtered) == 0


def test_apply_filter_no_constraints_returns_all(sample_chunks):
    f = MetadataFilter()
    filtered = apply_filter(sample_chunks, f)
    # All chunks pass min_chars=50 check since sample chunks are > 40 chars
    assert len(filtered) == len(sample_chunks)


def test_infer_filter_amar_question():
    f = infer_filter_from_question("Apa amar putusan dalam perkara ini?")
    assert f.sections is not None
    assert "amar_putusan" in f.sections or "konklusi" in f.sections


def test_infer_filter_pertimbangan_question():
    f = infer_filter_from_question("Apa alasan hukum Mahkamah dalam mempertimbangkan perkara?")
    assert f.sections is not None
    assert any("pertimbangan" in s for s in f.sections)


def test_infer_filter_pemohon_question():
    f = infer_filter_from_question("Siapa pemohon dalam perkara ini?")
    assert f.sections is not None
    assert any(s in f.sections for s in ["duduk_perkara", "pembukaan"])


def test_infer_filter_default_no_special_keywords():
    f = infer_filter_from_question("Apa kesimpulan dokumen ini?")
    # Default: no section whitelist, but has min_chars
    assert f.sections is None
    assert f.min_chars > 0
