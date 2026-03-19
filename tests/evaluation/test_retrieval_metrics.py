"""Tests for src.evaluation.retrieval_metrics"""

import pytest
from src.evaluation.retrieval_metrics import (
    compute_context_precision,
    compute_context_recall,
    compute_retrieval_metrics,
    is_relevant,
)


# ── is_relevant ───────────────────────────────────────────────────────────────

def test_identical_texts_are_relevant():
    text = "Mahkamah menolak permohonan pemohon untuk seluruhnya."
    assert is_relevant(text, text, threshold=0.5) is True


def test_completely_different_texts_not_relevant():
    chunk = "Pasal 28 UUD 1945 menjamin kebebasan berserikat."
    gold  = "Pemohon mengajukan bukti berupa akta notaris."
    assert is_relevant(chunk, gold, threshold=0.5) is False


def test_partial_overlap_above_threshold():
    chunk = "Mahkamah Konstitusi berwenang mengadili perkara pengujian undang-undang."
    gold  = "Mahkamah berwenang mengadili perkara konstitusi undang-undang dasar."
    # Shares: Mahkamah, berwenang, mengadili, perkara, undang-undang
    result = is_relevant(chunk, gold, threshold=0.3)
    assert result is True


# ── context_precision ─────────────────────────────────────────────────────────

def test_precision_all_relevant():
    chunks = ["Menolak permohonan pemohon.", "Pemohon memiliki kedudukan hukum."]
    gold   = ["Menolak permohonan pemohon."]
    p = compute_context_precision(chunks, gold, threshold=0.5)
    assert p > 0.0


def test_precision_empty_chunks():
    assert compute_context_precision([], ["some gold"], threshold=0.5) == 0.0


def test_precision_range():
    chunks = ["text a", "text b", "text c"]
    gold   = ["text a"]
    p = compute_context_precision(chunks, gold)
    assert 0.0 <= p <= 1.0


# ── context_recall ────────────────────────────────────────────────────────────

def test_recall_full_coverage():
    gold   = ["Mahkamah menolak permohonan."]
    chunks = ["Mahkamah menolak permohonan untuk seluruhnya."]
    r = compute_context_recall(chunks, gold, threshold=0.3)
    assert r == 1.0


def test_recall_no_coverage():
    gold   = ["pasal 28 undang-undang dasar"]
    chunks = ["cuaca hari ini sangat panas di Jakarta"]
    r = compute_context_recall(chunks, gold, threshold=0.5)
    assert r == 0.0


def test_recall_empty_gold():
    assert compute_context_recall(["some chunk"], [], threshold=0.5) == 0.0


# ── combined ─────────────────────────────────────────────────────────────────

def test_combined_metrics_returns_both_keys():
    result = compute_retrieval_metrics(["chunk"], ["gold"])
    assert "context_precision" in result
    assert "context_recall" in result
