"""
Knowledge Update Scenario (Section 9.3)
  - 3 new MK verdicts introduced AFTER initial system setup
  - Measures: update latency per architecture + QA performance on new verdicts
  - Tests: which architecture supports lower-friction corpus updates

Architecture update procedures:
  LC:           No update needed — inject at query time. Update latency = 0s.
  Simple RAG:   Chunk → embed → add to FAISS index. Measure time.
  Advanced RAG: Same as Simple RAG + BM25 index rebuild. Measure time.

Run:
  python experiments/run_knowledge_update.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from experiments.pipeline.runner import run_experiment
from src.data.cleaner import clean_file
from src.data.section_extractor import extract_and_save
from src.indexing.chunkers.fixed_size import FixedSizeChunker
from src.indexing.vector_store import VerdictIndexRegistry
from src.systems.advanced_rag.pipeline import AdvancedRAGSystem, AblationFlags
from src.systems.long_context import LongContextSystem
from src.systems.simple_rag import SimpleRAGSystem
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

# The 3 new verdicts must be placed in data/raw/txt/ and data/raw/json/
# before running this experiment. They are NOT in the original sample_50.
NEW_VERDICT_IDS = [
    # Populated after selecting 3 hold-out verdicts from the 887 corpus
    # e.g. "45_PUU-XX_2022", "12_PHPU-XXI_2023", "7_SKLN-XIX_2021"
]


def measure_update_latency(
    verdict_id: str,
    registry: VerdictIndexRegistry,
) -> dict:
    """
    Measure how long it takes to add a new verdict to the RAG index.

    Returns timing breakdown for each indexing step.
    """
    raw_txt  = cfg.paths.raw_txt  / f"{verdict_id}.txt"
    clean_dst = cfg.paths.cleaned / f"{verdict_id}.txt"
    sect_dst  = cfg.paths.sectioned

    timings = {}

    # Step 1: Clean text
    t0 = time.perf_counter()
    clean_file(raw_txt, clean_dst)
    timings["clean_s"] = round(time.perf_counter() - t0, 3)

    # Step 2: Extract sections
    t0 = time.perf_counter()
    extract_and_save(clean_dst, sect_dst)
    timings["section_extract_s"] = round(time.perf_counter() - t0, 3)

    # Step 3: Chunk + embed + index
    t0 = time.perf_counter()
    text   = clean_dst.read_text(encoding="utf-8")
    chunks = FixedSizeChunker().split(text, verdict_id=verdict_id)
    registry.build_and_save(verdict_id, chunks)
    timings["index_build_s"] = round(time.perf_counter() - t0, 3)

    timings["total_update_s"] = round(sum(timings.values()), 3)
    timings["verdict_id"] = verdict_id
    log.info(f"Update latency for {verdict_id}: {timings['total_update_s']}s total")
    return timings


def main(evaluate: bool = True) -> None:
    qa_path = cfg.paths.qa_dir / "qa_pairs_knowledge_update.jsonl"
    model   = cfg.models.phase1_model

    if not qa_path.exists():
        log.error(f"Knowledge update QA file not found: {qa_path}")
        return

    if not NEW_VERDICT_IDS:
        log.warning(
            "NEW_VERDICT_IDS is empty. "
            "Edit run_knowledge_update.py to add the 3 hold-out verdict IDs."
        )

    registry = VerdictIndexRegistry()
    update_timings = []

    # Measure update latency for RAG systems
    for verdict_id in NEW_VERDICT_IDS:
        timing = measure_update_latency(verdict_id, registry)
        update_timings.append(timing)

    if update_timings:
        timing_df = pd.DataFrame(update_timings)
        timing_df["lc_update_s"] = 0.0  # LC requires no update
        out = cfg.paths.results / "additional" / "knowledge_update" / "update_latency.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        timing_df.to_csv(out, index=False)
        log.info(f"\nUpdate Latency Summary:\n{timing_df.to_string(index=False)}")

    # QA performance on new verdicts
    systems = {
        "lc":           LongContextSystem(model=model),
        "simple_rag":   SimpleRAGSystem(model=model, registry=registry),
        "advanced_rag": AdvancedRAGSystem(
            model=model,
            flags=AblationFlags.full_advanced(),
            registry=registry,
        ),
    }

    for name, system in systems.items():
        log.info(f"\nKnowledge Update QA | condition={name}")
        results_dir = cfg.paths.results / "additional" / "knowledge_update" / name
        run_experiment(
            system=system,
            qa_path=qa_path,
            results_dir=results_dir,
            evaluate=evaluate,
        )

    log.info("\nKnowledge update experiment complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    main(evaluate=not args.no_eval)
