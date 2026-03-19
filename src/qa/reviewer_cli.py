"""
Human review CLI for QA draft validation.

Presents each draft QA pair for accept / modify / reject.
Accepted and modified pairs are written to qa_pairs_full.jsonl.

Run:
  python -m src.qa.reviewer_cli
  python -m src.qa.reviewer_cli --resume   # continue from last reviewed pair
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)
console = Console()

DRAFTS_PATH   = cfg.paths.qa_dir / "qa_drafts_raw.jsonl"
REVIEWED_PATH = cfg.paths.qa_dir / "qa_pairs_full.jsonl"
PROGRESS_PATH = cfg.paths.qa_dir / ".review_progress.json"


def _load_progress() -> set[str]:
    """Load already-reviewed question IDs."""
    if PROGRESS_PATH.exists():
        return set(json.loads(PROGRESS_PATH.read_text()))
    return set()


def _save_progress(reviewed_ids: set[str]) -> None:
    PROGRESS_PATH.write_text(json.dumps(list(reviewed_ids)))


def _display_pair(pair: dict, idx: int, total: int) -> None:
    console.print(f"\n[dim]── {idx+1}/{total} ─────────────────────────────────────[/dim]")
    console.print(Panel(
        f"[bold cyan]Verdict:[/] {pair['verdict_id']}\n"
        f"[bold cyan]Type:[/]    {pair['question_type']}\n"
        f"[bold cyan]ID:[/]      {pair['question_id']}",
        title="[bold]QA Pair",
        border_style="blue",
    ))
    console.print(Panel(pair["question"], title="[bold yellow]Question", border_style="yellow"))
    console.print(Panel(pair["gold_answer"], title="[bold green]Gold Answer", border_style="green"))
    if pair.get("gold_paragraphs"):
        console.print(Panel(
            "\n---\n".join(pair["gold_paragraphs"][:2]),
            title="[bold]Gold Paragraphs (supporting)",
            border_style="dim",
        ))


def run_review(resume: bool = False) -> None:
    if not DRAFTS_PATH.exists():
        console.print(f"[red]Draft file not found: {DRAFTS_PATH}[/red]")
        console.print("Run: bash scripts/generate_qa_drafts.sh")
        sys.exit(1)

    drafts = []
    with open(DRAFTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                drafts.append(json.loads(line))

    reviewed_ids = _load_progress() if resume else set()
    pending = [d for d in drafts if d["question_id"] not in reviewed_ids]

    console.print(f"\n[bold]QA Review CLI[/bold]")
    console.print(f"Total drafts: {len(drafts)} | Reviewed: {len(reviewed_ids)} | Pending: {len(pending)}")
    console.print("\n[dim]Commands: [a]ccept  [m]odify  [r]eject  [q]uit[/dim]\n")

    REVIEWED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(REVIEWED_PATH, "a", encoding="utf-8")

    accepted = modified = rejected = 0

    try:
        for i, pair in enumerate(pending):
            _display_pair(pair, i, len(pending))
            action = Prompt.ask(
                "[bold]Action[/bold]",
                choices=["a", "m", "r", "q"],
                default="a",
            )

            if action == "q":
                console.print("[yellow]Review paused. Run with --resume to continue.[/yellow]")
                break

            elif action == "a":
                pair["status"] = "accepted"
                out_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                out_file.flush()
                accepted += 1

            elif action == "m":
                new_q = Prompt.ask(
                    "New question (enter to keep)",
                    default=pair["question"],
                )
                new_a = Prompt.ask(
                    "New answer (enter to keep)",
                    default=pair["gold_answer"],
                )
                pair["question"]    = new_q
                pair["gold_answer"] = new_a
                pair["status"]      = "modified"
                out_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                out_file.flush()
                modified += 1

            elif action == "r":
                rejected += 1

            reviewed_ids.add(pair["question_id"])
            _save_progress(reviewed_ids)

    finally:
        out_file.close()

    console.print(f"\n[bold]Session summary:[/bold]")
    console.print(f"  Accepted:  {accepted}")
    console.print(f"  Modified:  {modified}")
    console.print(f"  Rejected:  {rejected}")
    console.print(f"  Output → {REVIEWED_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip already-reviewed pairs")
    args = parser.parse_args()
    run_review(resume=args.resume)
