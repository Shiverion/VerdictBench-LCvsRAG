"""
Phase 1: Controlled Architectural Comparison
  - Long Context vs Simple RAG vs Advanced RAG
  - Single model (Gemini 2.5 Flash), full 300 QA pairs

Run:
  python experiments/run_phase1.py
  python experiments/run_phase1.py --no-eval     # skip faithfulness (faster)
  python experiments/run_phase1.py --condition lc  # single condition only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.pipeline.runner import run_experiment
from src.systems.advanced_rag.pipeline import AdvancedRAGSystem, AblationFlags
from src.systems.long_context import LongContextSystem
from src.systems.simple_rag import SimpleRAGSystem
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


def main(condition: str | None = None, evaluate: bool = True, resume: bool = False) -> None:
    sample_meta = pd.read_csv(cfg.paths.sample_50) if cfg.paths.sample_50.exists() else None
    qa_path     = cfg.paths.qa_dir / "qa_pairs_full.jsonl"
    model       = cfg.models.phase1_model

    systems = {
        "lc": LongContextSystem(model=model),
        "simple_rag": SimpleRAGSystem(model=model),
        "advanced_rag": AdvancedRAGSystem(
            model=model,
            flags=AblationFlags.full_advanced(),
        ),
    }

    targets = {condition: systems[condition]} if condition else systems

    for name, system in targets.items():
        log.info(f"\n{'='*60}")
        log.info(f"Running condition: {name}")
        log.info(f"{'='*60}")
        results_dir = cfg.paths.results / "phase1" / name
        run_experiment(
            system=system,
            qa_path=qa_path,
            results_dir=results_dir,
            evaluate=evaluate,
            sample_metadata=sample_meta,
            resume=resume,
        )

    log.info("\nPhase 1 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 experiments")
    parser.add_argument("--condition", choices=["lc", "simple_rag", "advanced_rag"],
                        default=None, help="Run a single condition only")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed question_ids and append to existing results")
    parser.add_argument("--budget-idr", type=float, default=None,
                        help="Override max_cost_idr for this run")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip faithfulness evaluation (faster)")
    args = parser.parse_args()

    if args.budget_idr:
        cfg.budget.max_cost_idr = args.budget_idr

    main(condition=args.condition, evaluate=not args.no_eval, resume=args.resume)
