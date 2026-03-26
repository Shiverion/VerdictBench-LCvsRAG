"""
Inter-Annotator Agreement (IAA) calculator for QA dataset validation.

Supports:
- Cohen's Kappa (2 raters)
- Fleiss' Kappa (3+ raters via statsmodels)

Target: κ > 0.75
"""

import json
import argparse
import glob
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

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

def compute_multi_rater_iaa(file_paths: list[Path]) -> dict:
    """Compute agreement across multiple raters (Fleiss' Kappa)."""
    # Load all decisions
    all_decisions = [load_decisions(p) for p in file_paths]
    
    # Intersection of all question IDs
    common_ids = set.intersection(*(set(d.keys()) for d in all_decisions))
    if not common_ids:
        log.error("No overlapping question IDs found between all raters.")
        return {}

    # Category map for Fleiss' Kappa table
    # Categories: accepted, modified, rejected
    categories = ["accepted", "modified", "rejected"]
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    # Build Fleiss' Kappa count matrix: [n_items, n_categories]
    matrix = np.zeros((len(common_ids), len(categories)))
    
    for i, qid in enumerate(common_ids):
        for d in all_decisions:
            status = d[qid]
            if status in cat_to_idx:
                matrix[i, cat_to_idx[status]] += 1
            else:
                # Default to rejected if unknown status
                matrix[i, cat_to_idx["rejected"]] += 1

    kappa = fleiss_kappa(matrix)
    
    # Simple agreement rate
    agreement_rate = 0
    for row in matrix:
        if np.max(row) == len(all_decisions):
            agreement_rate += 1
    rate = agreement_rate / len(common_ids)

    result = {
        "kappa": round(kappa, 4),
        "n_pairs": len(common_ids),
        "n_raters": len(all_decisions),
        "agreement_rate": round(rate, 4),
        "method": "Fleiss' Kappa",
        "meets_target": kappa >= 0.75
    }

    log.info(f"Multi-Rater IAA Results ({result['method']}):")
    log.info(f"  n_pairs:        {result['n_pairs']}")
    log.info(f"  n_raters:       {result['n_raters']}")
    log.info(f"  agreement_rate: {result['agreement_rate']:.2%}")
    log.info(f"  Kappa:          {result['kappa']:.4f}")
    log.info(f"  Target (≥0.75): {'✓ MET' if result['meets_target'] else '✗ NOT MET'}")

    return result

def compute_pairwise_iaa(file1: Path, file2: Path) -> dict:
    """Cohen's Kappa for exactly 2 raters."""
    d1 = load_decisions(file1)
    d2 = load_decisions(file2)
    
    common_ids = list(set(d1.keys()) & set(d2.keys()))
    if not common_ids:
        log.error("No overlapping question IDs.")
        return {}

    labels1 = [d1[qid] for qid in common_ids]
    labels2 = [d2[qid] for qid in common_ids]
    kappa = cohen_kappa_score(labels1, labels2)
    rate = sum(1 for a, b in zip(labels1, labels2) if a == b) / len(common_ids)

    result = {
        "kappa": round(float(kappa), 4),
        "n_pairs": len(common_ids),
        "agreement_rate": round(float(rate), 4),
        "method": "Cohen's Kappa",
        "meets_target": kappa >= 0.75
    }

    log.info(f"Pairwise IAA Results ({result['method']}):")
    log.info(f"  n_pairs:        {result['n_pairs']}")
    log.info(f"  agreement_rate: {result['agreement_rate']:.2%}")
    log.info(f"  Kappa:          {result['kappa']:.4f}")
    log.info(f"  Target (>=0.75): {'PASSED' if result['meets_target'] else 'FAILED'}")

    return result

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--annotator2", help="Path to second annotator's JSONL file (Cohen's Kappa)")
    group.add_argument("--multi", help="Glob pattern for multiple annotator files (Fleiss' Kappa)")
    args = parser.parse_args()

    from src.utils.config import cfg
    primary_path = cfg.paths.qa_dir / "qa_pairs_full.jsonl"

    if args.multi:
        file_paths = [Path(p) for p in glob.glob(args.multi)]
        if primary_path.exists():
            file_paths = [primary_path] + file_paths
        
        # Unique paths only
        file_paths = list(dict.fromkeys(file_paths))
        if len(file_paths) < 2:
            log.error("Need at least 2 files for multi-rater IAA.")
            return
        compute_multi_rater_iaa(file_paths)
    else:
        # Cohen's Kappa for exactly 2 raters
        compute_pairwise_iaa(primary_path, Path(args.annotator2))

if __name__ == "__main__":
    main()
