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
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.cost_tracker import CostRecord, CostTracker, CostMeter, check_budget, CircuitBreaker, BudgetExceeded
from src.systems.base import QASystem, QAResult
from src.utils.config import cfg
from src.utils.logger import RunLogger, get_logger
from src.utils.token_counter import count_tokens, estimate_cost

log = get_logger(__name__)


def get_completed_ids(results_dir: Path) -> set:
    """Scans existing result JSONL files for a condition and returns the set of question_ids already processed.
    
    Skips records with errors or failed faithfulness evaluation (all statements
    have 'evaluation error' in reason), so resume mode will retry them.
    """
    completed = set()
    for jsonl_file in results_dir.glob("run_*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Skip records with generation errors
                    if record.get("error"):
                        continue
                    # Skip records where faithfulness eval failed
                    details = record.get("statement_details", [])
                    if details and all(
                        "evaluation error" in d.get("reason", "")
                        for d in details
                    ):
                        continue
                    completed.add(record["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


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
    resume: bool = False,
) -> Path:
    """
    Run a QA system against a dataset and write results to JSONL.

    Args:
        system:          Instantiated QASystem (LC, SimpleRAG, AdvancedRAG).
        qa_path:         Path to QA JSONL file.
        results_dir:     Directory to write run_YYYYMMDD_HHMMSS.jsonl.
        evaluate:        If True, run faithfulness + retrieval metrics per query.
        sample_metadata: DataFrame with verdict stratum info (for cost breakdown).
        resume:          If True, skips already completed queries and appends to existing JSONL.

    Returns:
        Path to the written JSONL file.
    """
    qa_pairs = load_qa_dataset(qa_path)
    
    if resume:
        completed_ids = get_completed_ids(results_dir)
        original_count = len(qa_pairs)
        qa_pairs = [p for p in qa_pairs if p.get("question_id") not in completed_ids]
        
        log.info(
            f"[{system.condition_name}] Resume mode: "
            f"{len(completed_ids)} already done, "
            f"{len(qa_pairs)} remaining / {original_count} total"
        )
        
        if not qa_pairs:
            log.info(f"[{system.condition_name}] Nothing to do — all {original_count} pairs complete.")
            return results_dir
    else:
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

    # Fast-fail budget guard
    check_budget(
        n_pairs=len(qa_pairs),
        gen_model=system.model,
        judge_model=cfg.models.judge_model,
        budget_idr=60_000
    )

    cost_tracker = CostTracker()
    gen_meter = CostMeter(model=system.model, batch_api=getattr(cfg.models, 'use_batch_api', False))
    judge_meter = CostMeter(model=cfg.models.judge_model, batch_api=getattr(cfg.models, 'use_batch_api', False))

    breaker = CircuitBreaker(
        max_input_tokens  = cfg.budget.max_input_tokens,
        max_output_tokens = cfg.budget.max_output_tokens,
        max_cost_usd      = cfg.budget.max_cost_usd,
        max_cost_idr      = cfg.budget.max_cost_idr,
        warn_at_pct       = cfg.budget.warn_at_pct,
    )

    with RunLogger(results_dir, system.condition_name, system.model, append=resume) as run_log:
        try:
            pbar = tqdm(qa_pairs, desc=f"[{system.condition_name}]")
            for i, pair in enumerate(pbar):
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
                judge_costs = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
                if evaluate:
                    # Faithfulness + hallucination
                    faith_result = evaluate_faithfulness(
                        answer=result.answer,
                        context=result.context_used,
                    )

                    # Extract usage from evaluation API responses
                    u = faith_result.pop("usage", {"input_tokens": 0, "output_tokens": 0})
                    judge_costs = judge_meter.record(
                        input_tokens=u.get("input_tokens", 0),
                        output_tokens=u.get("output_tokens", 0)
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
                gen_costs = gen_meter.record(
                    input_tokens=result.input_tokens,
                    # Assume output tokens roughly based on character count / 4 for simplicity if not tracked deeply
                    output_tokens=len(result.answer) // 4
                )

                # Additional detailed token fields requested by user
                record["gen_input_tokens"]    = gen_costs["input_tokens"]
                record["gen_output_tokens"]   = gen_costs["output_tokens"]
                record["gen_cost_usd"]        = gen_costs["cost_usd"]
                record["judge_input_tokens"]  = judge_costs["input_tokens"]
                record["judge_output_tokens"] = judge_costs["output_tokens"]
                record["judge_cost_usd"]      = judge_costs["cost_usd"]
                record["total_cost_usd"]      = gen_costs["cost_usd"] + judge_costs["cost_usd"]

                cost_tracker.add(CostRecord(
                    question_id=question_id,
                    verdict_id=verdict_id,
                    condition=system.condition_name,
                    model=system.model,
                    input_tokens=result.input_tokens,
                    cost_usd=gen_costs["cost_usd"],
                    stratum=stratum_lookup.get(verdict_id, ""),
                ))

                run_log.log(**record)

                breaker.check(gen_meter, judge_meter, label=f"{system.condition_name}[{i+1}]")

                total_usd = gen_meter.summary()["total_cost_usd"] + judge_meter.summary()["total_cost_usd"]
                pbar.set_postfix({
                    "Tokens": f"{gen_meter.summary()['total_input_tokens'] + judge_meter.summary()['total_input_tokens'] + gen_meter.summary()['total_output_tokens'] + judge_meter.summary()['total_output_tokens']:,}",
                    "Cost": f"Rp{total_usd * 16_000:,.0f}"
                })

        except BudgetExceeded as e:
            log.error(f"Run stopped early! Partial results preserved. Exception: {e}")

    # Save cost summaries
    gen_meter.log_summary(label=f"{system.condition_name}/generator")
    judge_meter.log_summary(label=f"{system.condition_name}/judge")

    cost_path = results_dir / f"cost_summary_{system.condition_name}.csv"
    cost_tracker.summary().to_csv(cost_path, mode='a' if resume else 'w', header=not (resume and cost_path.exists()), index=False)
    
    summary_path = results_dir / f"cost_meter_{system.condition_name}.json"
    new_gen = gen_meter.summary()
    new_judge = judge_meter.summary()

    if resume and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        def merge(existing_summary, new_summary):
            return {
                **new_summary,
                "total_calls":         existing_summary["total_calls"]         + new_summary["total_calls"],
                "total_input_tokens":  existing_summary["total_input_tokens"]  + new_summary["total_input_tokens"],
                "total_output_tokens": existing_summary["total_output_tokens"] + new_summary["total_output_tokens"],
                "total_cost_usd":      round(existing_summary["total_cost_usd"] + new_summary["total_cost_usd"], 4),
            }
        merged = {
            "generator": merge(existing["generator"], new_gen),
            "judge":     merge(existing["judge"],     new_judge),
        }
    else:
        merged = {"generator": new_gen, "judge": new_judge}

    summary_path.write_text(json.dumps(merged, indent=2))
    log.info(f"Detailed CostMeter summary → {summary_path}")

    return run_log.path
