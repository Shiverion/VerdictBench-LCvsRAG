"""
Inter-Annotator Agreement calculator for QA dataset validation.

Uses Cohen's Kappa on a 10% random sample reviewed by a second annotator.
Target: κ > 0.75

Second annotator reviews the same drafts independently using reviewer_cli.py
with a separate output file, then this script computes agreement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.utils.logger import get_logger

log = get_logger(__name__)


def load_decisions(jsonl_path: Path) -> dict[str, str]:
    """Load {question_id: status} from a reviewed JSONL file."""
    decisions = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                decisions[r["question_id"]] = r.get("status", "rejected")
    return decisions


def compute_kappa(
    annotator1_path: Path,
    annotator2_path: Path,
    sample_fraction: float = 0.10,
) -> dict:
    """
    Compute Cohen's Kappa between two annotators.

    Args:
        annotator1_path: JSONL from first reviewer (primary).
        annotator2_path: JSONL from second reviewer (IAA sample).
        sample_fraction: Fraction of pairs reviewed by both.

    Returns:
        Dict with kappa, n_pairs, agreement_rate, and per-class counts.
    """
    d1 = load_decisions(annotator1_path)
    d2 = load_decisions(annotator2_path)

    # Common question IDs only
    common_ids = set(d1.keys()) & set(d2.keys())
    if not common_ids:
        log.error("No overlapping question IDs found between annotators.")
        return {}

    labels1 = [d1[qid] for qid in common_ids]
    labels2 = [d2[qid] for qid in common_ids]

    kappa = cohen_kappa_score(labels1, labels2)

    # Agreement breakdown
    agree = sum(1 for a, b in zip(labels1, labels2) if a == b)
    rate  = agree / len(labels1)

    result = {
        "kappa":          round(kappa, 4),
        "n_pairs":        len(common_ids),
        "agreement_rate": round(rate, 4),
        "target_kappa":   0.75,
        "meets_target":   kappa >= 0.75,
    }

    log.info(f"IAA Results:")
    log.info(f"  n_pairs:        {result['n_pairs']}")
    log.info(f"  agreement_rate: {result['agreement_rate']:.2%}")
    log.info(f"  Cohen's Kappa:  {result['kappa']:.4f}")
    log.info(f"  Target (≥0.75): {'✓ MET' if result['meets_target'] else '✗ NOT MET'}")

    if kappa < 0.75:
        log.warning(
            "Kappa below target. Review disagreements and consider revising "
            "annotation guidelines before proceeding."
        )

    return result


if __name__ == "__main__":
    import argparse
    from src.utils.config import cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator2", required=True,
                        help="Path to second annotator's JSONL file")
    args = parser.parse_args()

    compute_kappa(
        annotator1_path=cfg.paths.qa_dir / "qa_pairs_full.jsonl",
        annotator2_path=Path(args.annotator2),
    )
