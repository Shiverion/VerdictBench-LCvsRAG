"""
Batch evaluator — post-hoc evaluation on completed JSONL result files.

Useful for:
  - Re-running evaluation with a different judge model
  - Adding BERTScore (expensive) after a quick first pass
  - Adding human legal accuracy scores back into the result file
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.evaluation.bertscore_eval import evaluate_bertscore
from src.evaluation.faithfulness import evaluate_faithfulness
from src.evaluation.hallucination import compute_hallucination
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.utils.logger import get_logger, load_run

log = get_logger(__name__)


def add_bertscore(results_path: Path, out_path: Path | None = None) -> pd.DataFrame:
    """
    Add BERTScore F1 to an existing result JSONL.

    Args:
        results_path: Path to run_*.jsonl.
        out_path:     Output path (defaults to results_path with _bs suffix).

    Returns:
        DataFrame with bertscore_f1 column added.
    """
    records = load_run(results_path)
    log.info(f"Adding BERTScore to {len(records)} records from {results_path.name}")

    predictions = [r.get("answer", "")     for r in records]
    references  = [r.get("gold_answer", "") for r in records]

    scores = evaluate_bertscore(predictions, references)

    for record, score in zip(records, scores):
        record["bertscore_f1"] = score

    out_path = out_path or results_path.with_stem(results_path.stem + "_bs")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(f"BERTScore results → {out_path}")
    return pd.DataFrame(records)


def re_evaluate_faithfulness(
    results_path: Path,
    out_path: Path | None = None,
    judge_model: str | None = None,
) -> pd.DataFrame:
    """
    Re-run faithfulness evaluation with a (possibly different) judge model.
    Overwrites faithfulness, hallucination_rate, hallucination_flag fields.
    """
    records = load_run(results_path)
    log.info(f"Re-evaluating faithfulness for {len(records)} records")

    for record in tqdm(records, desc="Re-evaluating"):
        faith = evaluate_faithfulness(
            answer=record.get("answer", ""),
            context=record.get("context_used", ""),
            judge_model=judge_model,
        )
        hall = compute_hallucination(faith["faithfulness"])
        record.update(faith)
        record.update(hall)

    out_path = out_path or results_path.with_stem(results_path.stem + "_refaith")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(f"Re-evaluated results → {out_path}")
    return pd.DataFrame(records)


def compile_summary(results_dir: Path) -> pd.DataFrame:
    """
    Compile all JSONL files under results_dir into a summary DataFrame.
    One row per condition with mean metrics.
    """
    all_records = []
    for p in sorted(results_dir.rglob("run_*.jsonl")):
        all_records.extend(load_run(p))

    if not all_records:
        log.warning(f"No result files found under {results_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    metric_cols = [
        "faithfulness", "hallucination_rate", "bertscore_f1",
        "context_precision", "context_recall",
        "input_tokens", "cost_usd", "latency_s",
    ]
    available = [c for c in metric_cols if c in df.columns]

    summary = (
        df.groupby(["condition", "model"])[available]
        .agg(["mean", "std", "count"])
        .round(4)
    )

    log.info(f"Compiled summary from {len(df)} records across {df.condition.nunique()} conditions")
    return summary
