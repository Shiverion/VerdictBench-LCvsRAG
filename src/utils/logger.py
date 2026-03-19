"""
Structured JSONL run logger.
Every QA evaluation result is written as one JSON line:
  { question_id, condition, model, answer, tokens_in, cost_usd, latency_s,
    faithfulness, hallucination_rate, bertscore_f1, ... }
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Standard Python logger with structured console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class RunLogger:
    """
    Writes one JSONL file per experiment run.
    File name: results/<condition>/run_YYYYMMDD_HHMMSS.jsonl

    Usage:
        logger = RunLogger(results_dir / "lc", condition="lc", model="gemini-2.5-flash")
        logger.log(question_id="q001", answer="...", tokens_in=4200, ...)
        logger.close()
    """

    def __init__(self, output_dir: Path, condition: str, model: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.path = output_dir / f"run_{ts}.jsonl"
        self._file = self.path.open("w", encoding="utf-8")
        self._meta = {"condition": condition, "model": model, "started_at": ts}
        self._count = 0
        self._logger = get_logger(f"RunLogger:{condition}")
        self._logger.info(f"Run log → {self.path}")

    def log(self, **kwargs: Any) -> None:
        record = {**self._meta, **kwargs,
                  "logged_at": datetime.now(timezone.utc).isoformat()}
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        self._count += 1

    def close(self) -> None:
        self._logger.info(f"Run complete — {self._count} records written to {self.path}")
        self._file.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def load_run(path: Path) -> list[dict]:
    """Load a JSONL result file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_runs(results_dir: Path) -> list[dict]:
    """Load all JSONL files under a results directory recursively."""
    records = []
    for p in sorted(results_dir.rglob("run_*.jsonl")):
        records.extend(load_run(p))
    return records
