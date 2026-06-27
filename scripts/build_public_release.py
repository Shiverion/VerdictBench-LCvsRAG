"""Build sanitized public release artifacts from private local results.

This script intentionally excludes raw QA text, answers, retrieved chunks,
gold paragraphs, context, and judge statement details.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PUBLIC_ROOT = Path("public_release")
AGG_DIR = PUBLIC_ROOT / "aggregate_results"
MKRI_PUTUSAN_PORTAL = "https://www.mkri.id/index.php?page=web.Putusan"

SENSITIVE_FIELDS = {
    "answer",
    "context_used",
    "gold_answer",
    "gold_evidence_statement_details",
    "gold_paragraphs",
    "question",
    "retrieved_chunks",
    "statement_details",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def infer_family(path: Path) -> str:
    parts = path.parts
    if "phase1" in parts:
        return "phase1"
    if "phase2" in parts:
        return "phase2"
    if "ablation" in parts:
        return "ablation"
    if "niah" in parts:
        return "niah"
    if "knowledge_update" in parts:
        return "knowledge_update"
    if "chunking_comparison" in parts:
        return "chunking_comparison"
    return "other"


def roman_year(file_id: str) -> int | None:
    match = re.search(r"_(X{0,3}(?:IX|IV|V?I{0,3}))_(\d{4})$", file_id)
    if match:
        return int(match.group(2))
    match = re.search(r"_(\d{4})$", file_id)
    if match:
        return int(match.group(1))
    return None


def safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 6)


def safe_sum(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.sum()), 6)


def build_verdict_manifest() -> None:
    source = Path("data/metadata/sample_50.csv")
    if not source.exists():
        return

    df = pd.read_csv(source)
    keep = pd.DataFrame(
        {
            "verdict_id": df["file_id"],
            "nomor_putusan": df.get("nomor_putusan"),
            "year": df["file_id"].map(roman_year),
            "stratum": df.get("stratum"),
            "n_chars": df.get("n_chars"),
            "est_pages": df.get("est_pages"),
            "sample_rank": df.get("sample_rank"),
            "official_source_url": "",
            "official_source_note": (
                "Search the public MKRI Putusan portal by verdict_id or nomor_putusan."
            ),
            "official_portal": MKRI_PUTUSAN_PORTAL,
        }
    )
    keep.to_csv(PUBLIC_ROOT / "verdict_manifest.csv", index=False)


def build_aggregate_results() -> None:
    files = sorted(
        p
        for p in Path("results_corrected/gold_evidence_faithfulness").rglob("*.jsonl")
        if "_sample" not in p.name
    )
    records: list[dict[str, Any]] = []
    for path in files:
        for row in read_jsonl(path):
            if SENSITIVE_FIELDS.intersection(row):
                row = {k: v for k, v in row.items() if k not in SENSITIVE_FIELDS}
            row["family"] = infer_family(path)
            row["source_artifact"] = path.as_posix()
            records.append(row)

    if not records:
        return

    df = pd.DataFrame(records)
    group_cols = [c for c in ["family", "model", "condition"] if c in df.columns]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out = dict(zip(group_cols, keys, strict=False))
        out["n_records"] = int(len(group))
        if "answer" in group.columns:
            out["n_answered"] = int(group["answer"].fillna("").astype(str).str.strip().ne("").sum())
        metric_cols = [
            "faithfulness",
            "gold_evidence_faithfulness",
            "gold_evidence_hallucination_rate",
            "hallucination_rate",
            "bertscore_f1",
            "context_precision",
            "context_recall",
            "legal_accuracy",
            "latency_s",
            "input_tokens",
            "gen_input_tokens",
            "gen_output_tokens",
        ]
        for col in metric_cols:
            if col in group.columns:
                out[f"mean_{col}"] = safe_mean(group[col])
        for col in ["cost_usd", "total_cost_usd", "gen_cost_usd", "judge_cost_usd"]:
            if col in group.columns:
                out[f"sum_{col}"] = safe_sum(group[col])
                out[f"mean_{col}"] = safe_mean(group[col])
        rows.append(out)

    summary = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    summary.to_csv(AGG_DIR / "gold_evidence_summary_by_condition.csv", index=False)

    if "question_type" in df.columns:
        qt_rows = []
        for keys, group in df.groupby(group_cols + ["question_type"], dropna=False):
            out = dict(zip(group_cols + ["question_type"], keys, strict=False))
            out["n_records"] = int(len(group))
            if "gold_evidence_faithfulness" in group.columns:
                out["mean_gold_evidence_faithfulness"] = safe_mean(
                    group["gold_evidence_faithfulness"]
                )
            qt_rows.append(out)
        pd.DataFrame(qt_rows).sort_values(group_cols + ["question_type"]).to_csv(
            AGG_DIR / "gold_evidence_summary_by_question_type.csv", index=False
        )


def build_redacted_sample() -> None:
    sample_path = PUBLIC_ROOT / "sample_redacted_qa.jsonl"
    sample_path.write_text(
        json.dumps(
            {
                "note": (
                    "No raw QA examples are redistributed in the public release because "
                    "the reviewed QA/evidence rows can contain personal data from court "
                    "documents. Add manually reviewed redacted examples here only."
                ),
                "status": "withheld_pending_manual_redaction",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    build_verdict_manifest()
    build_aggregate_results()
    build_redacted_sample()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
