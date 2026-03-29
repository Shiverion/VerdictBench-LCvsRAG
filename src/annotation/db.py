from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.utils.config import ROOT


DEFAULT_DB_PATH = Path(os.getenv("ANNOTATION_DB_PATH", ROOT / "data/qa_dataset/annotation_app.sqlite3"))


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is None:
        return Path(DEFAULT_DB_PATH)
    return Path(db_path)


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(db_path: str | Path | None = None):
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS annotators (
            annotator_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS items (
            question_id TEXT PRIMARY KEY,
            verdict_id TEXT NOT NULL,
            question TEXT NOT NULL,
            gold_answer TEXT NOT NULL,
            gold_paragraphs_json TEXT NOT NULL,
            question_type TEXT NOT NULL,
            source_status TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS assignments (
            assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id TEXT NOT NULL REFERENCES items(question_id) ON DELETE CASCADE,
            annotator_id TEXT NOT NULL REFERENCES annotators(annotator_id) ON DELETE CASCADE,
            assignment_order INTEGER NOT NULL,
            assigned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            UNIQUE(question_id, annotator_id)
        );

        CREATE TABLE IF NOT EXISTS annotations (
            annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL UNIQUE REFERENCES assignments(assignment_id) ON DELETE CASCADE,
            annotator_id TEXT NOT NULL REFERENCES annotators(annotator_id) ON DELETE CASCADE,
            question_id TEXT NOT NULL REFERENCES items(question_id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK (status IN ('accepted', 'modified', 'rejected')),
            edited_question TEXT,
            edited_gold_answer TEXT,
            edited_gold_paragraphs_json TEXT,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
