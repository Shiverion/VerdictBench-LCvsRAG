"""
Needle-in-a-Haystack (NIAH) Experiment
  - 30 questions where gold paragraph is in the bottom 20% of the document
  - Tests LC "lost in the middle" vulnerability vs RAG retrieval recall
  - Reports per-architecture NIAH accuracy vs full-corpus accuracy

Run:
  python experiments/run_niah.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.pipeline.runner import run_experiment
from src.systems.advanced_rag.pipeline import AdvancedRAGSystem, AblationFlags
from src.systems.long_context import LongContextSystem
from src.systems.simple_rag import SimpleRAGSystem
from src.utils.config import cfg
from src.utils.logger import get_logger, load_run

log = get_logger(__name__)


def compute_niah_accuracy(results_path: Path) -> dict:
    """
    Compute NIAH-specific accuracy from a result JSONL.

    A NIAH answer is considered correct if faithfulness >= 0.7.
    Reports both NIAH accuracy and comparison to full-corpus baseline.
    """
    records = load_run(results_path)
    if not records:
        return {}

    faithfulness_scores = [
        r.get("faithfulness", 0.0)
        for r in records
        if r.get("faithfulness") is not None
    ]

    if not faithfulness_scores:
        return {"n": len(records), "note": "faithfulness not evaluated"}

    niah_accuracy = sum(1 for s in faithfulness_scores if s >= 0.7) / len(faithfulness_scores)

    return {
        "condition":      records[0].get("condition"),
        "n_questions":    len(faithfulness_scores),
        "niah_accuracy":  round(niah_accuracy, 4),
        "mean_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
    }


def main(evaluate: bool = True) -> None:
    sample_meta = pd.read_csv(cfg.paths.sample_50) if cfg.paths.sample_50.exists() else None
    qa_path     = cfg.paths.qa_dir / "qa_pairs_niah.jsonl"
    model       = cfg.models.phase1_model

    if not qa_path.exists():
        log.error(f"NIAH QA file not found: {qa_path}")
        log.error("Build qa_pairs_niah.jsonl first (see data/qa_dataset/schema.md)")
        return

    systems = {
        "lc":           LongContextSystem(model=model),
        "simple_rag":   SimpleRAGSystem(model=model),
        "advanced_rag": AdvancedRAGSystem(model=model, flags=AblationFlags.full_advanced()),
    }

    niah_results = []
    for name, system in systems.items():
        log.info(f"\n{'='*60}")
        log.info(f"NIAH | condition={name}")
        log.info(f"{'='*60}")
        results_dir = cfg.paths.results / "additional" / "niah" / name
        result_path = run_experiment(
            system=system,
            qa_path=qa_path,
            results_dir=results_dir,
            evaluate=evaluate,
            sample_metadata=sample_meta,
        )
        if evaluate:
            niah_results.append(compute_niah_accuracy(result_path))

    if niah_results:
        summary = pd.DataFrame(niah_results)
        out = cfg.paths.results / "additional" / "niah" / "niah_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        log.info(f"\nNIAH Summary:\n{summary.to_string(index=False)}")
        log.info(f"Summary → {out}")

    log.info("\nNIAH experiment complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    main(evaluate=not args.no_eval)
