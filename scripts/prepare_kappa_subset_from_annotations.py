from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.annotation.db import connect, init_db


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a 2-annotator Cohen's kappa workspace from completed annotations."
    )
    parser.add_argument(
        "--source-db",
        default="data/qa_dataset/annotation_app.sqlite3",
        help="Source annotation SQLite database.",
    )
    parser.add_argument(
        "--target-db",
        default="data/qa_dataset/annotation_kappa.sqlite3",
        help="Target SQLite database for the kappa-only workspace.",
    )
    parser.add_argument(
        "--target-jsonl",
        default="data/qa_dataset/qa_pairs_iaa_from_annotator1.jsonl",
        help="Output JSONL containing the shared overlap items.",
    )
    parser.add_argument("--source-annotator", default="annotator_1")
    parser.add_argument("--peer-annotator", default="annotator_2")
    parser.add_argument("--n-items", type=int, default=30)
    args = parser.parse_args()

    source = connect(args.source_db)
    try:
        rows = source.execute(
            """
            SELECT
                i.question_id,
                i.verdict_id,
                i.question,
                i.gold_answer,
                i.gold_paragraphs_json,
                i.question_type,
                i.source_status,
                a.assignment_order,
                n.status,
                n.edited_question,
                n.edited_gold_answer,
                n.edited_gold_paragraphs_json,
                n.notes
            FROM annotations n
            JOIN assignments a ON a.assignment_id = n.assignment_id
            JOIN items i ON i.question_id = a.question_id
            WHERE a.annotator_id = ?
            ORDER BY a.assignment_order, i.question_id
            LIMIT ?
            """,
            (args.source_annotator, args.n_items),
        ).fetchall()
        if len(rows) < args.n_items:
            raise SystemExit(
                f"Need at least {args.n_items} completed items from {args.source_annotator}, found {len(rows)}."
            )
    finally:
        source.close()

    target_path = Path(args.target_db)
    if target_path.exists():
        target_path.unlink()

    target = connect(args.target_db)
    try:
        init_db(target)
        target.execute(
            "INSERT INTO annotators (annotator_id, display_name) VALUES (?, ?)",
            (args.source_annotator, args.source_annotator.replace("_", " ").title()),
        )
        target.execute(
            "INSERT INTO annotators (annotator_id, display_name) VALUES (?, ?)",
            (args.peer_annotator, args.peer_annotator.replace("_", " ").title()),
        )

        export_rows: list[dict] = []
        for order_idx, row in enumerate(rows, start=1):
            gold_paragraphs = json.loads(row["gold_paragraphs_json"])
            target.execute(
                """
                INSERT INTO items (
                    question_id, verdict_id, question, gold_answer, gold_paragraphs_json, question_type, source_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["question_id"],
                    row["verdict_id"],
                    row["question"],
                    row["gold_answer"],
                    row["gold_paragraphs_json"],
                    row["question_type"],
                    row["source_status"],
                ),
            )

            # Completed assignment for annotator_1
            target.execute(
                """
                INSERT INTO assignments (question_id, annotator_id, assignment_order, completed_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (row["question_id"], args.source_annotator, order_idx),
            )
            assignment_id = target.execute(
                """
                SELECT assignment_id FROM assignments
                WHERE question_id = ? AND annotator_id = ?
                """,
                (row["question_id"], args.source_annotator),
            ).fetchone()["assignment_id"]
            target.execute(
                """
                INSERT INTO annotations (
                    assignment_id, annotator_id, question_id, status,
                    edited_question, edited_gold_answer, edited_gold_paragraphs_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assignment_id,
                    args.source_annotator,
                    row["question_id"],
                    row["status"],
                    row["edited_question"],
                    row["edited_gold_answer"],
                    row["edited_gold_paragraphs_json"],
                    row["notes"],
                ),
            )

            # Pending assignment for annotator_2
            target.execute(
                """
                INSERT INTO assignments (question_id, annotator_id, assignment_order)
                VALUES (?, ?, ?)
                """,
                (row["question_id"], args.peer_annotator, order_idx),
            )

            export_rows.append(
                {
                    "question_id": row["question_id"],
                    "verdict_id": row["verdict_id"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "gold_paragraphs": gold_paragraphs,
                    "question_type": row["question_type"],
                    "status": row["source_status"] or "accepted",
                }
            )

        target.commit()
    finally:
        target.close()

    target_jsonl = Path(args.target_jsonl)
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)
    target_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in export_rows) + "\n",
        encoding="utf-8",
    )

    print(f"Created kappa DB: {Path(args.target_db)}")
    print(f"Created overlap JSONL: {target_jsonl}")
    print(f"Preserved {len(export_rows)} completed items from {args.source_annotator}")
    print(f"Pending on the same {len(export_rows)} items for {args.peer_annotator}")


if __name__ == "__main__":
    main()
