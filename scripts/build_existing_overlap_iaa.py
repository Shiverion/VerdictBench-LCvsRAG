from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


FULL_DATASET = Path("data/qa_dataset/qa_pairs_full.jsonl")
SOURCE_DB = Path("data/qa_dataset/annotation_app.sqlite3")
OUTPUT_DIR = Path("data/qa_dataset/iaa_existing_overlap")
SOURCE_ANNOTATOR_IN_DB = "annotator_1"


def _load_sqlite_rows() -> list[dict]:
    import sqlite3

    conn = sqlite3.connect(SOURCE_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                i.question_id,
                i.verdict_id,
                i.question,
                i.gold_answer,
                i.gold_paragraphs_json,
                i.question_type,
                n.status AS annotator2_status,
                n.edited_question,
                n.edited_gold_answer,
                n.edited_gold_paragraphs_json,
                n.notes,
                a.assignment_order
            FROM annotations n
            JOIN assignments a ON a.assignment_id = n.assignment_id
            JOIN items i ON i.question_id = a.question_id
            WHERE a.annotator_id = ?
            ORDER BY a.assignment_order, i.question_id
            """,
            (SOURCE_ANNOTATOR_IN_DB,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _summarize(rows_a: list[dict], rows_b: list[dict]) -> dict:
    labels_a = [row["status"] for row in rows_a]
    labels_b = [row["status"] for row in rows_b]
    agreement = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(labels_a) if labels_a else 0.0
    kappa = cohen_kappa_score(labels_a, labels_b) if labels_a else None
    if kappa != kappa:  # NaN
        kappa = None
    return {
        "n_pairs": len(labels_a),
        "agreement_rate": round(agreement, 4),
        "cohen_kappa": None if kappa is None else round(float(kappa), 4),
        "annotator1_status_counts": dict(Counter(labels_a)),
        "annotator2_status_counts": dict(Counter(labels_b)),
        "note": (
            "High raw agreement with low kappa can occur under strong class imbalance "
            "(for example, when almost all labels are 'accepted')."
        ),
    }


def main() -> None:
    full_rows = [
        json.loads(line)
        for line in FULL_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    full_by_id = {row["question_id"]: row for row in full_rows}
    app_rows = _load_sqlite_rows()

    overlap_ids = [row["question_id"] for row in app_rows if row["question_id"] in full_by_id]

    aligned_a34: list[dict] = []
    aligned_b34: list[dict] = []
    for row in app_rows:
        qid = row["question_id"]
        if qid not in full_by_id:
            continue
        gold = full_by_id[qid]
        aligned_a34.append(
            {
                "question_id": qid,
                "verdict_id": gold["verdict_id"],
                "question": gold["question"],
                "gold_answer": gold["gold_answer"],
                "gold_paragraphs": gold.get("gold_paragraphs", []),
                "question_type": gold.get("question_type"),
                "status": gold.get("status"),
                "annotator": "annotator_1",
                "source": "qa_pairs_full.jsonl",
            }
        )
        aligned_b34.append(
            {
                "question_id": qid,
                "verdict_id": row["verdict_id"],
                "question": row["edited_question"] or row["question"],
                "gold_answer": row["edited_gold_answer"] or row["gold_answer"],
                "gold_paragraphs": json.loads(row["edited_gold_paragraphs_json"])
                if row["edited_gold_paragraphs_json"]
                else json.loads(row["gold_paragraphs_json"]),
                "question_type": row["question_type"],
                "status": row["annotator2_status"],
                "annotator": "annotator_2",
                "source": "annotation_app.sqlite3",
                "notes": row["notes"],
            }
        )

    aligned_a30 = aligned_a34[:30]
    aligned_b30 = aligned_b34[:30]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUTPUT_DIR / "annotator1_overlap_34.jsonl", aligned_a34)
    _write_jsonl(OUTPUT_DIR / "annotator2_overlap_34.jsonl", aligned_b34)
    _write_jsonl(OUTPUT_DIR / "annotator1_overlap_30.jsonl", aligned_a30)
    _write_jsonl(OUTPUT_DIR / "annotator2_overlap_30.jsonl", aligned_b30)

    summary = {
        "overlap_34": _summarize(aligned_a34, aligned_b34),
        "overlap_30": _summarize(aligned_a30, aligned_b30),
        "overlap_question_ids_34": overlap_ids,
        "overlap_question_ids_30": overlap_ids[:30],
    }
    (OUTPUT_DIR / "kappa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote aligned overlap files to {OUTPUT_DIR}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
