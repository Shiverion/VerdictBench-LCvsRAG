"""
Human legal accuracy spot-check CLI.

Reviews a random sample of generated answers on a 0/1/2 scale:
  2 - Correct: factually correct and legally accurate
  1 - Partial: captures the main point but misses a legally relevant detail
  0 - Incorrect: factually wrong or legally misleading

Results are written to a companion JSONL file for downstream analysis
between automated faithfulness scores and human legal accuracy.

Run:
  python -m src.evaluation.legal_accuracy_cli --results results/phase1/lc/run_*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.utils.config import cfg
from src.utils.logger import get_logger, load_run

log = get_logger(__name__)
console = Console()


def sample_for_review(
    records: list[dict],
    fraction: float | None = None,
    sample_size: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Sample a fraction of records for human review."""
    if sample_size is not None:
        n = min(len(records), max(1, int(sample_size)))
    else:
        fraction = fraction or cfg.eval.human_spot_check_ratio
        n = max(1, int(len(records) * fraction))
    rng = random.Random(seed)
    return rng.sample(records, n)


def is_reviewable(record: dict) -> bool:
    """Keep only rows with an answer and a scored faithfulness value."""
    answer = record.get("answer")
    faithfulness = record.get("faithfulness")
    if answer is None:
        return False
    if isinstance(answer, str) and not answer.strip():
        return False
    if faithfulness is None:
        return False
    return True


def looks_truncated(record: dict) -> bool:
    """
    Heuristic truncation detector for obviously incomplete generations.

    Signals:
    - very short outputs
    - answer ending mid-word / without terminal punctuation
    - low generation token count on non-trivial questions
    """
    answer = (record.get("answer") or "").strip()
    if not answer:
        return False

    gen_output_tokens = record.get("gen_output_tokens")
    question_type = (record.get("question_type") or "").strip()

    if isinstance(gen_output_tokens, (int, float)) and gen_output_tokens <= 25:
        return True

    if len(answer) < 80 and not answer.endswith((".", "!", "?", '"', "'", "”", "’", ")", "]")):
        return True

    if question_type in {"multi_section_reasoning", "boundary", "structural"}:
        if len(answer) < 120 and not answer.endswith((".", "!", "?", '"', "'", "”", "’", ")", "]")):
            return True

    return False


def run_spot_check(
    results_path: Path,
    out_path: Path | None = None,
    sample_size: int | None = None,
    seed: int = 42,
    skip_truncated: bool = False,
) -> Path:
    """
    Interactive spot-check for a single result JSONL file.

    Args:
        results_path: Path to run_*.jsonl.
        out_path: Output path (defaults to results_path + _human.jsonl).
        sample_size: Optional exact number of responses to review.
        seed: Random seed for deterministic sampling.
        skip_truncated: If True, drop likely truncated generations before sampling.

    Returns:
        Path to output file with legal_accuracy scores added.
    """
    records = load_run(results_path)
    reviewable = [record for record in records if is_reviewable(record)]
    if skip_truncated:
        reviewable = [record for record in reviewable if not looks_truncated(record)]
    sample = sample_for_review(reviewable, sample_size=sample_size, seed=seed)
    out_path = out_path or results_path.with_stem(results_path.stem + "_human")

    console.print("\n[bold]Legal Accuracy Spot-Check[/bold]")
    console.print(f"Source file: {results_path.name}")
    console.print(f"Reviewable responses: {len(reviewable)}/{len(records)}")
    console.print(f"Reviewing: {len(sample)}")
    console.print(f"Skip likely truncated rows: {'yes' if skip_truncated else 'no'}")
    console.print("\n[dim]Scale: 2=Correct  1=Partial  0=Incorrect  q=quit[/dim]\n")

    reviewed: list[dict] = []

    for i, record in enumerate(sample, start=1):
        console.print(f"\n[dim]{'-' * 34} {i}/{len(sample)} {'-' * 34}[/dim]")
        console.print(
            Panel(
                f"[cyan]Verdict:[/] {record.get('verdict_id')}\n"
                f"[cyan]Type:[/] {record.get('question_type')}\n"
                f"[cyan]Condition:[/] {record.get('condition')}\n"
                f"[cyan]Faithfulness:[/] {record.get('faithfulness', 'N/A')}\n"
                f"[cyan]Gen output tokens:[/] {record.get('gen_output_tokens', 'N/A')}\n"
                f"[cyan]Possible truncation:[/] {'yes' if looks_truncated(record) else 'no'}",
                border_style="blue",
            )
        )
        console.print(Panel(record.get("question", ""), title="[yellow]Question"))
        console.print(Panel(record.get("gold_answer", ""), title="[green]Gold Answer"))
        console.print(Panel(record.get("answer", ""), title="[white]Generated Answer"))

        action = Prompt.ask("Score [0/1/2] or [q]uit", choices=["0", "1", "2", "q"])
        if action == "q":
            console.print("[yellow]Spot-check paused.[/yellow]")
            break

        reviewed_record = dict(record)
        reviewed_record["legal_accuracy"] = int(action)
        reviewed.append(reviewed_record)

    with out_path.open("w", encoding="utf-8") as handle:
        for reviewed_record in reviewed:
            handle.write(json.dumps(reviewed_record, ensure_ascii=False) + "\n")

    avg = sum(record["legal_accuracy"] for record in reviewed) / len(reviewed) if reviewed else 0.0
    console.print("\n[bold]Session complete:[/bold]")
    console.print(f"  Reviewed: {len(reviewed)}")
    console.print(f"  Mean legal accuracy: {avg:.2f} / 2.0")
    console.print(f"  Output -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        required=True,
        help="Path to result JSONL (supports glob, e.g. results/phase1/lc/run_*.jsonl)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Exact number of reviewable answers to score",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--skip-truncated",
        action="store_true",
        help="Exclude likely truncated generations before sampling",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.results))
    if not paths:
        log.error(f"No files found: {args.results}")
    else:
        run_spot_check(
            Path(paths[-1]),
            sample_size=args.sample_size,
            seed=args.seed,
            skip_truncated=args.skip_truncated,
        )
