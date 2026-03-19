"""
Hallucination rate computation.

HR = 1 - faithfulness  (at statement level)
Binary flag: hallucinated = True if HR > threshold (default 0.20)

High-stakes legal QA context: even 1-2 wrong statements can mislead
practitioners, so a 20% threshold is deliberately conservative.
"""

from __future__ import annotations

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_hallucination(
    faithfulness: float,
    threshold: float | None = None,
) -> dict:
    """
    Compute hallucination rate and binary flag from faithfulness score.

    Args:
        faithfulness: Faithfulness score (0–1) from evaluate_faithfulness().
        threshold:    HR > threshold → hallucination_flag = True.

    Returns:
        Dict with:
          - hallucination_rate  (float 0–1)
          - hallucination_flag  (bool)
    """
    threshold        = threshold if threshold is not None else cfg.eval.hallucination_threshold
    hallucination_rate = round(1.0 - faithfulness, 4)
    hallucination_flag = hallucination_rate > threshold

    if hallucination_flag:
        log.debug(
            f"Hallucination flagged: HR={hallucination_rate:.3f} > threshold={threshold}"
        )

    return {
        "hallucination_rate": hallucination_rate,
        "hallucination_flag": hallucination_flag,
    }
