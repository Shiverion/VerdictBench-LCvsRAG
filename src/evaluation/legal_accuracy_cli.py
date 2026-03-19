"""
Human legal accuracy spot-check CLI.

Reviews a 10% random sample of all generated answers on a 0/1/2 scale:
  2 — Correct: factually correct and legally accurate
  1 — Partial: captures main point but misses a legally relevant detail
  0 — Incorrect: factually wrong or legally misleading

Results are merged back into result JSONL files for correlation analysis
between automated faithfulness scores and human legal accuracy.

Run:
  python -m src.evaluation.legal_accuracy_cli --results results/phase1/lc/run_*.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt

from src.utils.config import cfg
from src.utils.logger import get_logger, load_run

log = get_logger(__name__)
console = Console()

SCORE_LABELS = {
    2: "[bold green]2 — Correct[/bold green]",
    1: "[bold yellow]1 — Partial[/bold yellow]",
    0: "[bold red]0 — Incorrect[/bold red]",
}


def sample_for_review(
    records: list[dict],
    fraction: float | None = None,
    seed: int = 42,
) -> list[dict]:
    """Sample a fraction of records for human review."""
    fraction = fraction or cfg.eval.human_spot_check_ratio
    n        = max(1, int(len(records) * fraction))
    rng      = random.Random(seed)
    return rng.sample(records, n)


def run_spot_check(results_path: Path, out_path: Path | None = None) -> Path:
    """
    Interactive spot-check for a single result JSONL file.

    Args:
        results_path: Path to run_*.jsonl.
        out_path:     Output path (defaults to results_path + _human.jsonl).

    Returns:
        Path to output file with legal_accuracy scores added.
    """
    records = load_run(results_path)
    sample  = sample_for_review(records)
    out_path = out_path or results_path.with_stem(results_path.stem + "_human")

    console.print(f"\n[bold]Legal Accuracy Spot-Check[/bold]")
    console.print(f"Reviewing {len(sample)}/{len(records)} responses from {results_path.name}")
    console.print(f"\n[dim]Scale: 2=Correct  1=Partial  0=Incorrect  q=quit[/dim]\n")

    reviewed = []

    for i, record in enumerate(sample):
        console.print(f"\n[dim]── {i+1}/{len(sample)} ──────────────────────────────────[/dim]")
        console.print(Panel(
            f"[cyan]Verdict:[/] {record.get('verdict_id')}\n"
            f"[cyan]Type:[/]    {record.get('question_type')}\n"
            f"[cyan]Condition:[/] {record.get('condition')}  "
            f"[cyan]Faithfulness:[/] {record.get('faithfulness', 'N/A')}",
            border_style="blue",
        ))
        console.print(Panel(record.get("question", ""), title="[yellow]Question"))
        console.print(Panel(record.get("gold_answer", ""), title="[green]Gold Answer"))
        console.print(Panel(record.get("answer", ""), title="[white]Generated Answer"))

        action = Prompt.ask("Score [0/1/2] or [q]uit", choices=["0","1","2","q"])
        if action == "q":
            console.print("[yellow]Spot-check paused.[/yellow]")
            break

        record["legal_accuracy"] = int(action)
        reviewed.append(record)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in reviewed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg = sum(r["legal_accuracy"] for r in reviewed) / len(reviewed) if reviewed else 0.0
    console.print(f"\n[bold]Session complete:[/bold]")
    console.print(f"  Reviewed: {len(reviewed)}")
    console.print(f"  Mean legal accuracy: {avg:.2f} / 2.0")
    console.print(f"  Output → {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to result JSONL (supports glob, e.g. results/phase1/lc/run_*.jsonl)")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.results))
    if not paths:
        log.error(f"No files found: {args.results}")
    else:
        run_spot_check(Path(paths[-1]))  # use most recent run
