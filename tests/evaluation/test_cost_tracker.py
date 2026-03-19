"""Tests for src.evaluation.cost_tracker and src.utils.token_counter"""

import pytest
from src.evaluation.cost_tracker import CostRecord, CostTracker
from src.utils.token_counter import count_tokens, estimate_cost


# ── token counting ────────────────────────────────────────────────────────────

def test_count_tokens_gemini_nonempty():
    text = "Mahkamah Konstitusi berwenang mengadili perkara."
    n = count_tokens(text, model="gemini-2.5-flash-preview-04-17")
    assert n > 0


def test_count_tokens_empty():
    assert count_tokens("", model="gemini-2.5-flash-preview-04-17") == 0


def test_count_tokens_scales_with_length():
    short = "a"
    long  = "a " * 1000
    n_short = count_tokens(short, "gemini-2.5-flash-preview-04-17")
    n_long  = count_tokens(long,  "gemini-2.5-flash-preview-04-17")
    assert n_long > n_short


# ── cost estimation ────────────────────────────────────────────────────────────

def test_estimate_cost_zero_tokens():
    assert estimate_cost(0, "gemini-2.5-flash-preview-04-17", {"gemini-2.5-flash-preview-04-17": 0.15}) == 0.0


def test_estimate_cost_known_model():
    pricing = {"gemini-2.5-flash-preview-04-17": 0.15}
    cost = estimate_cost(1_000_000, "gemini-2.5-flash-preview-04-17", pricing)
    assert abs(cost - 0.15) < 1e-9


def test_estimate_cost_unknown_model_returns_zero():
    cost = estimate_cost(1000, "unknown-model-xyz", {})
    assert cost == 0.0


# ── CostTracker ───────────────────────────────────────────────────────────────

def test_cost_tracker_summary():
    tracker = CostTracker()
    for i in range(5):
        tracker.add(CostRecord(
            question_id=f"q{i}",
            verdict_id="v1",
            condition="lc",
            model="gemini-2.5-flash-preview-04-17",
            input_tokens=1000 * (i + 1),
            cost_usd=0.001 * (i + 1),
        ))
    summary = tracker.summary()
    assert len(summary) == 1
    assert summary.iloc[0]["n_queries"] == 5
    assert summary.iloc[0]["total_tokens"] == 15000


def test_cost_tracker_empty():
    tracker = CostTracker()
    assert tracker.summary().empty
