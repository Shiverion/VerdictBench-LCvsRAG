"""
Phase 2: Context Window Size Interaction
  - 2x3 factorial: Gemini 2.5 Flash (1M ctx) vs GPT-4o (128k ctx)
  - x LC / Simple RAG / Advanced RAG
  - Full 300 QA pairs.
Run:
  python experiments/run_phase2.py
  python experiments/run_phase2.py --model gemini_flash  # one model only
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

MODEL_CONFIGS = {
    "gemini_flash": {
        "model":    cfg.models.phase2_model_a,
        "windowed": False,
    },
    "gpt4o": {
        "model":    cfg.models.phase2_model_b,
        "windowed": True,   # LC windowed for GPT-4o (128k limit)
    },
}


def main(model_key: str | None = None, evaluate: bool = True, resume: bool = False) -> None:
    sample_meta = pd.read_csv(cfg.paths.sample_50) if cfg.paths.sample_50.exists() else None
    qa_path     = cfg.paths.qa_dir / "qa_pairs_full.jsonl"

    model_targets = {model_key: MODEL_CONFIGS[model_key]} if model_key else MODEL_CONFIGS

    for m_key, m_cfg in model_targets.items():
        model    = m_cfg["model"]
        windowed = m_cfg["windowed"]

        systems = {
            "lc":           LongContextSystem(model=model, windowed=windowed),
            "simple_rag":   SimpleRAGSystem(model=model),
            "advanced_rag": AdvancedRAGSystem(model=model, flags=AblationFlags.full_advanced()),
        }

        for cond_name, system in systems.items():
            log.info(f"\n{'='*60}")
            log.info(f"Phase 2 | model={m_key} | condition={cond_name}")
            log.info(f"{'='*60}")
            results_dir = cfg.paths.results / "phase2" / m_key / cond_name
            run_experiment(
                system=system,
                qa_path=qa_path,
                results_dir=results_dir,
                evaluate=evaluate,
                sample_metadata=sample_meta,
                resume=resume,
            )

    log.info("\nPhase 2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 experiments")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                        default=None, help="Run a single model only")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed (non-error) question_ids")
    args = parser.parse_args()
    main(model_key=args.model, evaluate=not args.no_eval, resume=args.resume)
