"""
Section 8.4: Fixed-size vs Section-boundary Chunking Comparison

Hypothesis: Section-boundary chunking improves context precision on
multi-section reasoning questions by preventing chunk boundary artifacts
at MK paragraph ([X.X]) boundaries.

Conditions:
  A: Simple RAG + fixed-size chunking (512 tokens, recursive char splitter)
  B: Simple RAG + section-boundary chunking ([X.X] aligned)

Run:
  python experiments/run_chunking_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.pipeline.runner import run_experiment
from src.systems.simple_rag import SimpleRAGSystem
from src.utils.config import cfg
from src.utils.logger import get_logger, load_run

log = get_logger(__name__)


def compare_by_question_type(results_dir: Path) -> pd.DataFrame:
    """
    Compare context precision by question type for fixed vs section chunking.
    Key question: does section chunking help more on multi_section_reasoning type?
    """
    records = []
    for jsonl in results_dir.rglob("run_*.jsonl"):
        records.extend(load_run(jsonl))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "context_precision" not in df.columns or "question_type" not in df.columns:
        return df

    pivot = df.groupby(["condition", "question_type"])["context_precision"].mean().unstack()
    return pivot.round(4)


def main(evaluate: bool = True) -> None:
    sample_meta = pd.read_csv(cfg.paths.sample_50) if cfg.paths.sample_50.exists() else None
    qa_path     = cfg.paths.qa_dir / "qa_pairs_ablation.jsonl"
    model       = cfg.models.phase1_model
    base_dir    = cfg.paths.results / "additional" / "chunking_comparison"

    conditions = {
        "simple_rag_fixed":   SimpleRAGSystem(model=model, chunking_strategy="fixed"),
        "simple_rag_section": SimpleRAGSystem(model=model, chunking_strategy="section"),
    }

    for name, system in conditions.items():
        log.info(f"\n{'='*60}")
        log.info(f"Chunking Comparison | condition={name}")
        log.info(f"{'='*60}")
        run_experiment(
            system=system,
            qa_path=qa_path,
            results_dir=base_dir / name,
            evaluate=evaluate,
            sample_metadata=sample_meta,
        )

    # Post-hoc comparison by question type
    comparison = compare_by_question_type(base_dir)
    if not comparison.empty:
        out = base_dir / "comparison_by_question_type.csv"
        comparison.to_csv(out)
        log.info(f"\nContext Precision by Question Type:\n{comparison.to_string()}")
        log.info(f"Saved → {out}")

    log.info("\nChunking comparison complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    main(evaluate=not args.no_eval)
