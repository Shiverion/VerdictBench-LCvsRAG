"""
Input token cost tracking and reporting.

Cost model:
  - Only INPUT tokens are tracked as primary metric
  - Output tokens are logged as supplementary but excluded from primary cost
  - Rationale: input cost IS the architectural differentiator
    (LC pays for full doc, RAG pays for chunks only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CostRecord:
    question_id:  str
    verdict_id:   str
    condition:    str
    model:        str
    input_tokens: int
    cost_usd:     float
    stratum:      str = ""   # short / medium / long — populated from sample metadata


@dataclass
class CostTracker:
    """Accumulates cost records across an experiment run."""
    records: list[CostRecord] = field(default_factory=list)

    def add(self, record: CostRecord) -> None:
        self.records.append(record)

    def summary(self) -> pd.DataFrame:
        """Per-condition cost summary."""
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame([r.__dict__ for r in self.records])
        return (
            df.groupby(["condition", "model"])
            .agg(
                n_queries        =("question_id", "count"),
                total_tokens     =("input_tokens", "sum"),
                mean_tokens      =("input_tokens", "mean"),
                total_cost_usd   =("cost_usd", "sum"),
                mean_cost_usd    =("cost_usd", "mean"),
                cost_per_1k_queries=("cost_usd", lambda x: x.mean() * 1000),
            )
            .reset_index()
            .round(6)
        )

    def by_stratum(self) -> pd.DataFrame:
        """Cost breakdown by document length stratum."""
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame([r.__dict__ for r in self.records])
        if "stratum" not in df.columns or df.stratum.isna().all():
            return pd.DataFrame()
        return (
            df.groupby(["condition", "stratum"])
            .agg(
                mean_tokens   =("input_tokens", "mean"),
                mean_cost_usd =("cost_usd", "mean"),
            )
            .reset_index()
            .round(6)
        )

    def save(self, path: Path) -> None:
        pd.DataFrame([r.__dict__ for r in self.records]).to_csv(path, index=False)
        log.info(f"Cost records saved → {path}")


def cost_report(results_jsonl: Path) -> pd.DataFrame:
    """
    Generate cost summary from a completed results JSONL file.
    Useful for post-hoc analysis without re-running experiments.
    """
    import json
    records = []
    with open(results_jsonl) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records.append({
                    "condition":    r.get("condition"),
                    "model":        r.get("model"),
                    "input_tokens": r.get("input_tokens", 0),
                    "cost_usd":     r.get("cost_usd", 0.0),
                    "stratum":      r.get("stratum", ""),
                })
    df = pd.DataFrame(records)
    summary = df.groupby(["condition", "model"]).agg(
        n_queries=("input_tokens", "count"),
        total_tokens=("input_tokens", "sum"),
        mean_tokens=("input_tokens", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        mean_cost_usd=("cost_usd", "mean"),
    ).reset_index().round(6)
    return summary
