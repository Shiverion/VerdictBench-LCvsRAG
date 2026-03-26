"""
Re-evaluate faithfulness for existing Phase 2 result files.

Reads existing JSONL records that have valid answers but broken faithfulness
(evaluation error in statement_details), and re-runs ONLY the faithfulness
evaluation using the fixed two-model judge pipeline.

Usage:
  uv run python experiments/reeval_phase2.py
  uv run python experiments/reeval_phase2.py --model gemini_flash
  uv run python experiments/reeval_phase2.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.evaluation.faithfulness import evaluate_faithfulness
from src.evaluation.hallucination import compute_hallucination
from src.evaluation.cost_tracker import CostMeter
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


def needs_reeval(record: dict) -> bool:
    """Check if a record needs re-evaluation."""
    if record.get("error"):
        return False  # Skip error-only records (no answer to eval)
    details = record.get("statement_details", [])
    if not details:
        return True  # No eval was done
    return all("evaluation error" in d.get("reason", "") for d in details)


def reeval_file(jsonl_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Re-evaluate faithfulness for records in a single JSONL file.

    Returns (total_records, re_evaluated_count).
    """
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    to_reeval = [(i, r) for i, r in enumerate(records) if needs_reeval(r)]

    if not to_reeval:
        log.info(f"  {jsonl_path.name}: {len(records)} records, 0 need re-eval — skipping")
        return len(records), 0

    log.info(f"  {jsonl_path.name}: {len(records)} records, {len(to_reeval)} need re-eval")

    if dry_run:
        return len(records), len(to_reeval)

    judge_meter = CostMeter(model=cfg.models.judge_model, batch_api=False)

    for idx, record in tqdm(to_reeval, desc=f"  [{jsonl_path.stem}]"):
        answer = record.get("answer", "")
        context = record.get("context_used", "")

        faith_result = evaluate_faithfulness(
            answer=answer,
            context=context,
        )

        u = faith_result.pop("usage", {"input_tokens": 0, "output_tokens": 0})
        judge_costs = judge_meter.record(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
        )

        hall_result = compute_hallucination(faith_result["faithfulness"])

        # Update record in-place
        record.update(faith_result)
        record.update(hall_result)
        record["judge_input_tokens"] = judge_costs["input_tokens"]
        record["judge_output_tokens"] = judge_costs["output_tokens"]
        record["judge_cost_usd"] = judge_costs["cost_usd"]
        record["total_cost_usd"] = record.get("gen_cost_usd", 0) + judge_costs["cost_usd"]

    # Write back all records
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    judge_meter.log_summary(label=f"reeval/{jsonl_path.parent.name}")
    log.info(f"  ✓ Re-evaluated {len(to_reeval)} records, written back to {jsonl_path.name}")
    return len(records), len(to_reeval)


def main(model_key: str | None = None, dry_run: bool = False) -> None:
    base = cfg.paths.results / "phase2"
    total_records = 0
    total_reeval = 0

    model_dirs = [base / model_key] if model_key else sorted(base.iterdir())

    for model_dir in model_dirs:
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        for cond_dir in sorted(model_dir.iterdir()):
            if not cond_dir.is_dir() or cond_dir.name.startswith("."):
                continue

            log.info(f"\n{'='*60}")
            log.info(f"Re-eval | {model_dir.name}/{cond_dir.name}")
            log.info(f"{'='*60}")

            for jsonl_path in sorted(cond_dir.glob("run_*.jsonl")):
                n, r = reeval_file(jsonl_path, dry_run=dry_run)
                total_records += n
                total_reeval += r

    action = "Would re-evaluate" if dry_run else "Re-evaluated"
    log.info(f"\n{action} {total_reeval} / {total_records} total records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate faithfulness for Phase 2 results")
    parser.add_argument("--model", choices=["gemini_flash", "gpt4o"],
                        default=None, help="Re-eval a single model only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be re-evaluated without running")
    args = parser.parse_args()
    main(model_key=args.model, dry_run=args.dry_run)
