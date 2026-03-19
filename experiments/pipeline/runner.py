"""
Core experiment runner — loads a YAML config and runs a QA system
against a QA dataset, logging all results to JSONL.

This is the single execution engine called by all run_*.py scripts.
Adding a new experiment = new YAML config + new run_*.py that calls this.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.evaluation.faithfulness import evaluate_faithfulness
from src.evaluation.hallucination import compute_hallucination
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.cost_tracker import CostRecord, CostTracker
from src.systems.base import QASystem, QAResult
from src.utils.config import cfg
from src.utils.logger import RunLogger, get_logger
from src.utils.token_counter import count_tokens, estimate_cost

log = get_logger(__name__)


def load_qa_dataset(path: Path) -> list[dict]:
    """Load a QA JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_experiment(
    system: QASystem,
    qa_path: Path,
    results_dir: Path,
    evaluate: bool = True,
    sample_metadata: pd.DataFrame | None = None,
) -> Path:
    """
    Run a QA system against a dataset and write results to JSONL.

    Args:
        system:          Instantiated QASystem (LC, SimpleRAG, AdvancedRAG).
        qa_path:         Path to QA JSONL file.
        results_dir:     Directory to write run_YYYYMMDD_HHMMSS.jsonl.
        evaluate:        If True, run faithfulness + retrieval metrics per query.
        sample_metadata: DataFrame with verdict stratum info (for cost breakdown).

    Returns:
        Path to the written JSONL file.
    """
    qa_pairs = load_qa_dataset(qa_path)
    log.info(
        f"Running [{system.condition_name}] on {len(qa_pairs)} QA pairs "
        f"from {qa_path.name}"
    )

    # Build stratum lookup from sample metadata
    stratum_lookup: dict[str, str] = {}
    if sample_metadata is not None and "file_id" in sample_metadata.columns:
        stratum_lookup = dict(
            zip(sample_metadata.file_id, sample_metadata.get("stratum", [""] * len(sample_metadata)))
        )

    cost_tracker = CostTracker()

    with RunLogger(results_dir, system.condition_name, system.model) as run_log:
        for pair in tqdm(qa_pairs, desc=f"[{system.condition_name}]"):
            question_id    = pair.get("question_id", "")
            question       = pair.get("question", "")
            verdict_id     = pair.get("verdict_id", "")
            gold_answer    = pair.get("gold_answer", "")
            gold_paragraphs = pair.get("gold_paragraphs", [])
            question_type  = pair.get("question_type", "")

            try:
                result: QAResult = system.query(
                    question=question,
                    verdict_id=verdict_id,
                    question_id=question_id,
                )
            except Exception as e:
                log.error(f"Query failed for {question_id}: {e}")
                run_log.log(
                    question_id=question_id,
                    verdict_id=verdict_id,
                    error=str(e),
                )
                continue

            record = result.to_dict()
            record["gold_answer"]     = gold_answer
            record["gold_paragraphs"] = gold_paragraphs
            record["question_type"]   = question_type
            record["stratum"]         = stratum_lookup.get(verdict_id, "")

            # ── Evaluation ────────────────────────────────────────────────────
            if evaluate:
                # Faithfulness + hallucination
                faith_result = evaluate_faithfulness(
                    answer=result.answer,
                    context=result.context_used,
                )
                hall_result = compute_hallucination(faith_result["faithfulness"])
                record.update(faith_result)
                record.update(hall_result)

                # Retrieval metrics (RAG conditions only)
                if result.retrieved_chunks and gold_paragraphs:
                    ret_metrics = compute_retrieval_metrics(
                        result.retrieved_chunks, gold_paragraphs
                    )
                    record.update(ret_metrics)

            # ── Cost tracking ─────────────────────────────────────────────────
            cost_tracker.add(CostRecord(
                question_id=question_id,
                verdict_id=verdict_id,
                condition=system.condition_name,
                model=system.model,
                input_tokens=result.input_tokens,
                cost_usd=result.cost_usd,
                stratum=stratum_lookup.get(verdict_id, ""),
            ))

            run_log.log(**record)

    # Save cost summary alongside results
    cost_path = results_dir / f"cost_summary_{system.condition_name}.csv"
    cost_tracker.summary().to_csv(cost_path, index=False)
    log.info(f"Cost summary → {cost_path}")

    return run_log.path
