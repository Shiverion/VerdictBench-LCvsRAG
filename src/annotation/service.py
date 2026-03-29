from __future__ import annotations

import json
import math
import random
import sqlite3
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from src.annotation.db import init_db
from src.annotation.schemas import (
    AgreementSummary,
    AdminSummary,
    AnnotatorCreate,
    AnnotatorSummary,
    AssignmentView,
    BootstrapRequest,
)
from src.utils.config import ROOT


STATUS_ORDER = ["accepted", "modified", "rejected"]


def _normalize_dataset_path(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def ensure_ready(conn: sqlite3.Connection) -> None:
    init_db(conn)


def reset_annotation_data(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM annotations")
    conn.execute("DELETE FROM assignments")
    conn.execute("DELETE FROM items")
    conn.execute("DELETE FROM annotators")


def create_annotator(conn: sqlite3.Connection, annotator: AnnotatorCreate) -> None:
    conn.execute(
        """
        INSERT INTO annotators (annotator_id, display_name)
        VALUES (?, ?)
        ON CONFLICT(annotator_id) DO UPDATE SET display_name = excluded.display_name
        """,
        (annotator.annotator_id, annotator.display_name),
    )


def list_annotators(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT annotator_id, display_name, created_at FROM annotators ORDER BY annotator_id"
    ).fetchall()
    return [dict(row) for row in rows]


def import_items(conn: sqlite3.Connection, dataset_path: str) -> int:
    path = _normalize_dataset_path(dataset_path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        conn.execute(
            """
            INSERT INTO items (
                question_id, verdict_id, question, gold_answer, gold_paragraphs_json, question_type, source_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(question_id) DO UPDATE SET
                verdict_id = excluded.verdict_id,
                question = excluded.question,
                gold_answer = excluded.gold_answer,
                gold_paragraphs_json = excluded.gold_paragraphs_json,
                question_type = excluded.question_type,
                source_status = excluded.source_status
            """,
            (
                record["question_id"],
                record["verdict_id"],
                record["question"],
                record["gold_answer"],
                json.dumps(record.get("gold_paragraphs", []), ensure_ascii=False),
                record.get("question_type", "unknown"),
                record.get("status"),
            ),
        )
    return len(records)


def _choose_annotators_for_item(
    rng: random.Random,
    annotator_ids: list[str],
    load_by_annotator: dict[str, int],
    assignments_per_item: int,
) -> list[str]:
    shuffled = annotator_ids[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda annotator_id: (load_by_annotator[annotator_id], annotator_id))
    return shuffled[:assignments_per_item]


def generate_assignments(conn: sqlite3.Connection, req: BootstrapRequest) -> int:
    annotator_ids = req.annotator_ids
    if req.assignments_per_item > len(annotator_ids):
        raise ValueError("assignments_per_item cannot exceed number of annotators")

    rows = conn.execute("SELECT question_id FROM items ORDER BY question_id").fetchall()
    question_ids = [row["question_id"] for row in rows]
    existing = conn.execute("SELECT COUNT(*) AS count FROM assignments").fetchone()["count"]
    if existing:
        return existing

    rng = random.Random(req.seed)
    load_by_annotator = {annotator_id: 0 for annotator_id in annotator_ids}
    assignment_order = 1

    for question_id in question_ids:
        chosen = _choose_annotators_for_item(
            rng=rng,
            annotator_ids=annotator_ids,
            load_by_annotator=load_by_annotator,
            assignments_per_item=req.assignments_per_item,
        )
        for annotator_id in chosen:
            conn.execute(
                """
                INSERT INTO assignments (question_id, annotator_id, assignment_order)
                VALUES (?, ?, ?)
                """,
                (question_id, annotator_id, assignment_order),
            )
            load_by_annotator[annotator_id] += 1
        assignment_order += 1

    return len(question_ids) * req.assignments_per_item


def bootstrap_annotation_project(conn: sqlite3.Connection, req: BootstrapRequest) -> dict:
    ensure_ready(conn)
    if req.reset_existing:
        reset_annotation_data(conn)

    for annotator_id in req.annotator_ids:
        create_annotator(
            conn,
            AnnotatorCreate(
                annotator_id=annotator_id,
                display_name=annotator_id.replace("_", " ").title(),
            ),
        )

    imported = import_items(conn, req.dataset_path)
    assignments = generate_assignments(conn, req)
    return {
        "imported_items": imported,
        "total_assignments": assignments,
        "annotators": list_annotators(conn),
    }


def get_next_assignment(conn: sqlite3.Connection, annotator_id: str) -> AssignmentView | None:
    row = conn.execute(
        """
        SELECT
            a.assignment_id,
            a.annotator_id,
            i.question_id,
            i.verdict_id,
            i.question,
            i.gold_answer,
            i.gold_paragraphs_json,
            i.question_type,
            i.source_status
        FROM assignments a
        JOIN items i ON i.question_id = a.question_id
        LEFT JOIN annotations n ON n.assignment_id = a.assignment_id
        WHERE a.annotator_id = ? AND n.annotation_id IS NULL
        ORDER BY a.assignment_order, a.assignment_id
        LIMIT 1
        """,
        (annotator_id,),
    ).fetchone()
    if row is None:
        return None

    return AssignmentView(
        assignment_id=row["assignment_id"],
        annotator_id=row["annotator_id"],
        item={
            "question_id": row["question_id"],
            "verdict_id": row["verdict_id"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "gold_paragraphs": json.loads(row["gold_paragraphs_json"]),
            "question_type": row["question_type"],
            "source_status": row["source_status"],
        },
    )


def submit_annotation(
    conn: sqlite3.Connection,
    *,
    assignment_id: int,
    annotator_id: str,
    status: str,
    edited_question: str | None,
    edited_gold_answer: str | None,
    edited_gold_paragraphs: list[str] | None,
    notes: str | None,
) -> None:
    assignment = conn.execute(
        "SELECT assignment_id, annotator_id, question_id FROM assignments WHERE assignment_id = ?",
        (assignment_id,),
    ).fetchone()
    if assignment is None:
        raise ValueError("Assignment not found")
    if assignment["annotator_id"] != annotator_id:
        raise ValueError("Annotator does not own this assignment")

    conn.execute(
        """
        INSERT INTO annotations (
            assignment_id, annotator_id, question_id, status,
            edited_question, edited_gold_answer, edited_gold_paragraphs_json, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(assignment_id) DO UPDATE SET
            status = excluded.status,
            edited_question = excluded.edited_question,
            edited_gold_answer = excluded.edited_gold_answer,
            edited_gold_paragraphs_json = excluded.edited_gold_paragraphs_json,
            notes = excluded.notes,
            created_at = CURRENT_TIMESTAMP
        """,
        (
            assignment_id,
            annotator_id,
            assignment["question_id"],
            status,
            edited_question,
            edited_gold_answer,
            json.dumps(edited_gold_paragraphs, ensure_ascii=False) if edited_gold_paragraphs else None,
            notes,
        ),
    )
    conn.execute(
        "UPDATE assignments SET completed_at = CURRENT_TIMESTAMP WHERE assignment_id = ?",
        (assignment_id,),
    )


def _collect_agreement_items(conn: sqlite3.Connection) -> list[AgreementSummary]:
    rows = conn.execute(
        """
        SELECT
            i.question_id,
            i.verdict_id,
            n.status
        FROM items i
        JOIN assignments a ON a.question_id = i.question_id
        JOIN annotations n ON n.assignment_id = a.assignment_id
        ORDER BY i.question_id
        """
    ).fetchall()

    by_question: dict[str, dict] = {}
    for row in rows:
        bucket = by_question.setdefault(
            row["question_id"],
            {"verdict_id": row["verdict_id"], "votes": Counter()},
        )
        bucket["votes"][row["status"]] += 1

    results: list[AgreementSummary] = []
    for question_id, payload in by_question.items():
        votes = payload["votes"]
        ordered = votes.most_common()
        if not ordered:
            continue
        top_count = ordered[0][1]
        top_labels = sorted([label for label, count in ordered if count == top_count])
        results.append(
            AgreementSummary(
                question_id=question_id,
                verdict_id=payload["verdict_id"],
                votes={key: votes.get(key, 0) for key in STATUS_ORDER},
                consensus_status=top_labels[0] if len(top_labels) == 1 else None,
                is_tie=len(top_labels) > 1,
            )
        )
    return results


def _compute_cohen_kappa(conn: sqlite3.Connection) -> float | None:
    rows = conn.execute(
        """
        SELECT
            i.question_id,
            a.annotator_id,
            n.status
        FROM items i
        JOIN assignments a ON a.question_id = i.question_id
        JOIN annotations n ON n.assignment_id = a.assignment_id
        ORDER BY i.question_id, a.annotator_id
        """
    ).fetchall()
    if not rows:
        return None

    statuses_by_question: dict[str, dict[str, str]] = {}
    for row in rows:
        bucket = statuses_by_question.setdefault(row["question_id"], {})
        bucket[row["annotator_id"]] = row["status"]

    shared = [bucket for bucket in statuses_by_question.values() if len(bucket) >= 2]
    if not shared:
        return None

    labels_a: list[str] = []
    labels_b: list[str] = []
    for bucket in shared:
        ordered = sorted(bucket.items())
        labels_a.append(ordered[0][1])
        labels_b.append(ordered[1][1])
    kappa = float(cohen_kappa_score(labels_a, labels_b))
    if math.isnan(kappa):
        agreement = all(a == b for a, b in zip(labels_a, labels_b))
        return 1.0 if agreement else None
    return round(kappa, 4)


def build_admin_summary(conn: sqlite3.Connection) -> AdminSummary:
    annotator_rows = conn.execute(
        """
        SELECT
            an.annotator_id,
            an.display_name,
            COUNT(DISTINCT ass.assignment_id) AS assigned,
            COUNT(DISTINCT ann.annotation_id) AS completed
        FROM annotators an
        LEFT JOIN assignments ass ON ass.annotator_id = an.annotator_id
        LEFT JOIN annotations ann ON ann.assignment_id = ass.assignment_id
        GROUP BY an.annotator_id, an.display_name
        ORDER BY an.annotator_id
        """
    ).fetchall()
    annotators = [AnnotatorSummary(**dict(row)) for row in annotator_rows]

    status_counts = {
        row["status"]: row["count"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS count FROM annotations GROUP BY status ORDER BY status"
        ).fetchall()
    }

    total_items = conn.execute("SELECT COUNT(*) AS count FROM items").fetchone()["count"]
    total_assignments = conn.execute("SELECT COUNT(*) AS count FROM assignments").fetchone()["count"]
    completed_assignments = conn.execute(
        "SELECT COUNT(*) AS count FROM annotations"
    ).fetchone()["count"]
    completed_items = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM (
            SELECT a.question_id
            FROM assignments a
            JOIN annotations n ON n.assignment_id = a.assignment_id
            GROUP BY a.question_id
            HAVING COUNT(*) >= 2
        )
        """
    ).fetchone()["count"]

    return AdminSummary(
        total_items=total_items,
        total_assignments=total_assignments,
        completed_assignments=completed_assignments,
        completed_items=completed_items,
        cohen_kappa=_compute_cohen_kappa(conn),
        annotators=annotators,
        status_counts=status_counts,
        agreement_items=_collect_agreement_items(conn),
    )


def export_review_outputs(conn: sqlite3.Connection, output_dir: str) -> dict:
    target_dir = Path(output_dir)
    if not target_dir.is_absolute():
        target_dir = ROOT / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = conn.execute(
        """
        SELECT
            i.question_id,
            i.verdict_id,
            i.question,
            i.gold_answer,
            i.gold_paragraphs_json,
            i.question_type,
            i.source_status,
            a.annotator_id,
            n.status,
            n.edited_question,
            n.edited_gold_answer,
            n.edited_gold_paragraphs_json,
            n.notes
        FROM items i
        JOIN assignments ass ON ass.question_id = i.question_id
        JOIN annotations n ON n.assignment_id = ass.assignment_id
        JOIN annotators a ON a.annotator_id = ass.annotator_id
        ORDER BY i.question_id, a.annotator_id
        """
    ).fetchall()

    grouped: dict[str, dict] = {}
    for row in rows:
        item = grouped.setdefault(
            row["question_id"],
            {
                "question_id": row["question_id"],
                "verdict_id": row["verdict_id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "gold_paragraphs": json.loads(row["gold_paragraphs_json"]),
                "question_type": row["question_type"],
                "source_status": row["source_status"],
                "annotations": [],
            },
        )
        item["annotations"].append(
            {
                "annotator_id": row["annotator_id"],
                "status": row["status"],
                "edited_question": row["edited_question"],
                "edited_gold_answer": row["edited_gold_answer"],
                "edited_gold_paragraphs": json.loads(row["edited_gold_paragraphs_json"])
                if row["edited_gold_paragraphs_json"]
                else None,
                "notes": row["notes"],
            }
        )

    consensus_records: list[dict] = []
    disagreement_records: list[dict] = []
    for record in grouped.values():
        votes = Counter(annotation["status"] for annotation in record["annotations"])
        top = votes.most_common()
        if len(record["annotations"]) < 2:
            disagreement_records.append({**record, "votes": dict(votes), "consensus_status": None})
            continue
        top_count = top[0][1]
        top_labels = sorted(label for label, count in top if count == top_count)
        consensus_status = top_labels[0] if len(top_labels) == 1 else None
        payload = {**record, "votes": dict(votes), "consensus_status": consensus_status}
        if consensus_status is None:
            disagreement_records.append(payload)
        else:
            consensus_records.append(payload)

    consensus_path = target_dir / "consensus.jsonl"
    disagreement_path = target_dir / "disagreements.jsonl"
    consensus_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in consensus_records) + ("\n" if consensus_records else ""),
        encoding="utf-8",
    )
    disagreement_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in disagreement_records)
        + ("\n" if disagreement_records else ""),
        encoding="utf-8",
    )

    return {
        "consensus_path": str(consensus_path),
        "disagreements_path": str(disagreement_path),
        "consensus_count": len(consensus_records),
        "disagreement_count": len(disagreement_records),
    }
