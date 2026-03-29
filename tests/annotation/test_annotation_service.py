from __future__ import annotations

import json

from src.annotation.db import init_db
from src.annotation.schemas import BootstrapRequest
from src.annotation.service import (
    bootstrap_annotation_project,
    build_admin_summary,
    export_review_outputs,
    get_next_assignment,
    submit_annotation,
)


def _write_dataset(path, count: int = 4) -> None:
    records = []
    for idx in range(count):
        records.append(
            {
                "question_id": f"q{idx}",
                "verdict_id": f"v{idx % 2}",
                "question": f"Question {idx}",
                "gold_answer": f"Answer {idx}",
                "gold_paragraphs": [f"Paragraph {idx}"],
                "question_type": "factual_extractive",
                "status": "accepted",
            }
        )
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def test_bootstrap_creates_three_assignments_per_item(tmp_path, sqlite_connection):
    dataset = tmp_path / "qa.jsonl"
    _write_dataset(dataset)

    result = bootstrap_annotation_project(
        sqlite_connection,
        BootstrapRequest(
            dataset_path=str(dataset),
            annotator_ids=["ann_a", "ann_b"],
            assignments_per_item=2,
            reset_existing=True,
        ),
    )

    assert result["imported_items"] == 4
    total_assignments = sqlite_connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
    assert total_assignments == 8

    assignments_per_item = sqlite_connection.execute(
        "SELECT question_id, COUNT(*) FROM assignments GROUP BY question_id"
    ).fetchall()
    assert all(row[1] == 2 for row in assignments_per_item)


def test_summary_reports_agreement_and_cohen_kappa(tmp_path, sqlite_connection):
    dataset = tmp_path / "qa.jsonl"
    _write_dataset(dataset, count=1)

    bootstrap_annotation_project(
        sqlite_connection,
        BootstrapRequest(
            dataset_path=str(dataset),
            annotator_ids=["ann_a", "ann_b"],
            assignments_per_item=2,
            reset_existing=True,
        ),
    )

    for annotator_id, status in [
        ("ann_a", "accepted"),
        ("ann_b", "accepted"),
    ]:
        assignment = get_next_assignment(sqlite_connection, annotator_id)
        assert assignment is not None
        submit_annotation(
            sqlite_connection,
            assignment_id=assignment.assignment_id,
            annotator_id=annotator_id,
            status=status,
            edited_question=None,
            edited_gold_answer=None,
            edited_gold_paragraphs=None,
            notes=None,
        )

    summary = build_admin_summary(sqlite_connection)

    assert summary.completed_items == 1
    assert summary.cohen_kappa is not None
    assert summary.agreement_items[0].consensus_status == "accepted"


def test_export_writes_consensus_and_disagreement_files(tmp_path, sqlite_connection):
    dataset = tmp_path / "qa.jsonl"
    _write_dataset(dataset, count=2)

    bootstrap_annotation_project(
        sqlite_connection,
        BootstrapRequest(
            dataset_path=str(dataset),
            annotator_ids=["ann_a", "ann_b"],
            assignments_per_item=2,
            reset_existing=True,
        ),
    )

    # q0 consensus
    for annotator_id, status in [("ann_a", "accepted"), ("ann_b", "accepted")]:
        assignment = get_next_assignment(sqlite_connection, annotator_id)
        submit_annotation(
            sqlite_connection,
            assignment_id=assignment.assignment_id,
            annotator_id=annotator_id,
            status=status,
            edited_question=None,
            edited_gold_answer=None,
            edited_gold_paragraphs=None,
            notes=None,
        )

    # q1 disagreement
    for annotator_id, status in [("ann_a", "modified"), ("ann_b", "rejected")]:
        assignment = get_next_assignment(sqlite_connection, annotator_id)
        submit_annotation(
            sqlite_connection,
            assignment_id=assignment.assignment_id,
            annotator_id=annotator_id,
            status=status,
            edited_question=None,
            edited_gold_answer=None,
            edited_gold_paragraphs=None,
            notes=None,
        )

    out_dir = tmp_path / "exports"
    result = export_review_outputs(sqlite_connection, str(out_dir))
    assert result["consensus_count"] == 1
    assert result["disagreement_count"] == 1
    assert (out_dir / "consensus.jsonl").exists()
    assert (out_dir / "disagreements.jsonl").exists()
