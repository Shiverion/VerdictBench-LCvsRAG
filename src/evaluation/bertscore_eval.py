"""
BERTScore F1 evaluation using IndoBERT (or multilingual BERT fallback).

Indonesian legal text requires a multilingual or Indonesian-specific model
to avoid English-centric token bias from standard bert-base-uncased.

Models tried (in order of preference):
  1. indolem/indobert-base-uncased   — trained on Indonesian text
  2. bert-base-multilingual-cased    — fallback (mBERT)
"""

from __future__ import annotations

import os
from functools import lru_cache

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.utils.logger import get_logger
log = get_logger(__name__)

from transformers import AutoTokenizer
orig_from_pretrained = AutoTokenizer.from_pretrained

def patched_from_pretrained(*args, **kwargs):
    tokenizer = orig_from_pretrained(*args, **kwargs)
    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length > 1_000_000:
        log.info(f"Patching insane model_max_length: {tokenizer.model_max_length} -> 512")
        tokenizer.model_max_length = 512
    return tokenizer

AutoTokenizer.from_pretrained = patched_from_pretrained

from bert_score import score as bert_score_fn

# Indonesian BERT model for legal text evaluation
_PRIMARY_MODEL   = "indolem/indobert-base-uncased"
_FALLBACK_MODEL  = "bert-base-multilingual-cased"


@lru_cache(maxsize=1)
def _get_model() -> str:
    """Return best available model name."""
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(_PRIMARY_MODEL)
        log.info(f"BERTScore using primary model: {_PRIMARY_MODEL}")
        return _PRIMARY_MODEL
    except Exception:
        log.warning(f"Primary model unavailable, falling back to {_FALLBACK_MODEL}")
        return _FALLBACK_MODEL


def evaluate_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str | None = None,
    batch_size: int = 32,
) -> list[float]:
    """
    Compute BERTScore F1 for a batch of prediction-reference pairs.

    Args:
        predictions: List of generated answers.
        references:  List of ground truth answers.
        model_type:  BERT model name (defaults to IndoBERT).
        batch_size:  Batch size for inference.

    Returns:
        List of F1 scores (one per pair), range [0, 1].
    """
    assert len(predictions) == len(references), "predictions and references must be same length"

    if not predictions:
        return []

    model_type = model_type or _get_model()

    _, _, F1 = bert_score_fn(
        cands=predictions,
        refs=references,
        model_type=model_type,
        num_layers=12,
        lang="id",
        batch_size=batch_size,
        verbose=False,
    )

    scores = F1.tolist()
    avg = sum(scores) / len(scores) if scores else 0.0
    log.debug(f"BERTScore F1: mean={avg:.4f} over {len(scores)} pairs")
    return [round(s, 4) for s in scores]


def evaluate_single_bertscore(
    prediction: str,
    reference: str,
    model_type: str | None = None,
) -> float:
    """Convenience wrapper for a single prediction-reference pair."""
    results = evaluate_bertscore([prediction], [reference], model_type=model_type)
    return results[0] if results else 0.0
