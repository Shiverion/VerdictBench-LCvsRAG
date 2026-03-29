from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.annotation.db import session
from src.annotation.service import build_admin_summary, export_review_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export annotation consensus files and print IAA summary.")
    parser.add_argument(
        "--output-dir",
        default="data/qa_dataset/annotation_exports",
        help="Directory for consensus/disagreement JSONL exports.",
    )
    args = parser.parse_args()

    with session() as conn:
        summary = build_admin_summary(conn)
        export_result = export_review_outputs(conn, args.output_dir)

    line = {
        "n_items": summary.completed_items,
        "n_assignments": summary.completed_assignments,
        "cohen_kappa": summary.cohen_kappa,
        "status_counts": summary.status_counts,
        "consensus_count": export_result["consensus_count"],
        "disagreement_count": export_result["disagreement_count"],
        "consensus_path": export_result["consensus_path"],
        "disagreements_path": export_result["disagreements_path"],
    }

    print(json.dumps(line, ensure_ascii=False, indent=2))
    print()
    print(
        "Paper-ready summary:",
        f"On {summary.completed_items} shared QA items reviewed by two annotators,",
        f"Cohen's kappa was {summary.cohen_kappa}.",
    )


if __name__ == "__main__":
    main()
