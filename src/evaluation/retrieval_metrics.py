"""
Context Precision and Context Recall for RAG evaluation.

Gold supporting paragraphs are identified during QA dataset construction
and stored as `gold_paragraphs` in each QA pair.

Relevance判定: Jaccard token overlap ≥ threshold (default 0.50).

Context Precision = (retrieved chunks containing gold context) / (total retrieved chunks)
Context Recall    = (gold paragraphs covered by retrieved chunks) / (total gold paragraphs)
"""

from __future__ import annotations

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


def _tokenize(text: str) -> set[str]:
    """Simple whitespace tokenizer for Jaccard similarity."""
    return set(text.lower().split())


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two text strings."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    intersection = ta & tb
    union        = ta | tb
    return len(intersection) / len(union)


def is_relevant(chunk_text: str, gold_paragraph: str, threshold: float | None = None) -> bool:
    """
    Determine if a retrieved chunk covers a gold supporting paragraph.

    A chunk is considered relevant if its Jaccard similarity with the
    gold paragraph exceeds the threshold.

    Args:
        chunk_text:     Text of the retrieved chunk.
        gold_paragraph: Ground truth supporting paragraph.
        threshold:      Jaccard similarity threshold (default from config).

    Returns:
        True if the chunk covers the gold paragraph.
    """
    threshold = threshold if threshold is not None else cfg.eval.jaccard_threshold
    return _jaccard(chunk_text, gold_paragraph) >= threshold


def compute_context_precision(
    retrieved_chunks: list[str],
    gold_paragraphs: list[str],
    threshold: float | None = None,
) -> float:
    """
    Context Precision: fraction of retrieved chunks that contain gold context.

    Args:
        retrieved_chunks: Texts of retrieved chunks (from QAResult.retrieved_chunks).
        gold_paragraphs:  Ground truth supporting paragraphs for this question.
        threshold:        Jaccard threshold for relevance.

    Returns:
        Precision score [0, 1]. Returns 0.0 if no chunks retrieved.
    """
    if not retrieved_chunks:
        return 0.0

    relevant_count = 0
    for chunk in retrieved_chunks:
        # A chunk is relevant if it covers ANY gold paragraph
        if any(is_relevant(chunk, g, threshold) for g in gold_paragraphs):
            relevant_count += 1

    precision = relevant_count / len(retrieved_chunks)
    return round(precision, 4)


def compute_context_recall(
    retrieved_chunks: list[str],
    gold_paragraphs: list[str],
    threshold: float | None = None,
) -> float:
    """
    Context Recall: fraction of gold paragraphs covered by retrieved chunks.

    Args:
        retrieved_chunks: Texts of retrieved chunks.
        gold_paragraphs:  Ground truth supporting paragraphs.
        threshold:        Jaccard threshold for coverage.

    Returns:
        Recall score [0, 1]. Returns 1.0 for LC (full doc = all gold covered).
        Returns 0.0 if no gold paragraphs defined.
    """
    if not gold_paragraphs:
        return 0.0

    # For LC: full document is injected, so by definition all gold paragraphs
    # should be present. This is handled by the caller passing the full text
    # as a single "chunk" — recall will approach 1.0.

    covered = 0
    for gold in gold_paragraphs:
        if any(is_relevant(chunk, gold, threshold) for chunk in retrieved_chunks):
            covered += 1

    recall = covered / len(gold_paragraphs)
    return round(recall, 4)


def compute_retrieval_metrics(
    retrieved_chunks: list[str],
    gold_paragraphs: list[str],
    threshold: float | None = None,
) -> dict:
    """
    Compute both context precision and recall in one call.

    Returns:
        Dict with context_precision and context_recall.
    """
    return {
        "context_precision": compute_context_precision(
            retrieved_chunks, gold_paragraphs, threshold
        ),
        "context_recall": compute_context_recall(
            retrieved_chunks, gold_paragraphs, threshold
        ),
    }
