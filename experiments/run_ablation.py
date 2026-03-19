"""
Advanced RAG Component Ablation Study
  - 6 conditions on 100-question subset
  - Isolates contribution of each component via incremental attribution

Run:
  python experiments/run_ablation.py
  python experiments/run_ablation.py --condition plus_reranking
"""

from __future__ import annotations

import argparse

import pandas as pd

from experiments.pipeline.runner import run_experiment
from src.systems.advanced_rag.pipeline import AdvancedRAGSystem, AblationFlags
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

ABLATION_CONDITIONS: dict[str, AblationFlags] = {
    "simple_rag_baseline": AblationFlags.simple_rag(),
    "plus_query_rewrite":  AblationFlags.query_rewrite_only(),
    "plus_reranking":      AblationFlags.reranking_only(),
    "plus_hybrid_search":  AblationFlags.hybrid_only(),
    "plus_metadata_filter":AblationFlags.metadata_only(),
    "full_advanced_rag":   AblationFlags.full_advanced(),
}


def main(condition: str | None = None, evaluate: bool = True) -> None:
    sample_meta = pd.read_csv(cfg.paths.sample_50) if cfg.paths.sample_50.exists() else None
    qa_path     = cfg.paths.qa_dir / "qa_pairs_ablation.jsonl"
    model       = cfg.models.phase1_model

    targets = {condition: ABLATION_CONDITIONS[condition]} if condition else ABLATION_CONDITIONS

    for cond_name, flags in targets.items():
        log.info(f"\n{'='*60}")
        log.info(f"Ablation | condition={cond_name} | flags={flags}")
        log.info(f"{'='*60}")
        system = AdvancedRAGSystem(model=model, flags=flags)
        results_dir = cfg.paths.results / "ablation" / cond_name
        run_experiment(
            system=system,
            qa_path=qa_path,
            results_dir=results_dir,
            evaluate=evaluate,
            sample_metadata=sample_meta,
        )

    log.info("\nAblation study complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Advanced RAG ablation study")
    parser.add_argument("--condition", choices=list(ABLATION_CONDITIONS.keys()),
                        default=None)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    main(condition=args.condition, evaluate=not args.no_eval)
