"""Privacy audit for public VerdictBench artifacts.

The scanner reports counts only. It intentionally never prints matched text.
Use it before publishing:

    uv run python scripts/pii_audit.py --tracked --fail-on-signals --fail-on-jsonl-free-text
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_SCAN_PREFIXES = (
    "data/",
    "notebooks/",
    "output/",
    "public_release/",
    "results/",
    "results_corrected/",
)

SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}

TEXT_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".json",
    ".jsonl",
    ".md",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

PII_PATTERNS = {
    "possible_email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "possible_phone_or_id": re.compile(r"\b(?:\+62|62|0)8[0-9][0-9\s\-.]{7,}\b"),
    "nik_or_ktp_keyword": re.compile(
        r"\b(?:NIK|Nomor Induk Kependudukan|KTP|Kartu Tanda Penduduk)\b", re.I
    ),
    "address_keyword": re.compile(
        r"\b(?:alamat|beralamat|tempat tinggal|domisili|jalan|jl\.)\b", re.I
    ),
    "occupation_keyword": re.compile(
        r"\b(?:pekerjaan|wiraswasta|pegawai|karyawan|advokat|notaris|dosen|"
        r"mahasiswa|pensiunan)\b",
        re.I,
    ),
}

DISALLOWED_JSONL_FIELDS = {
    "answer",
    "context_used",
    "gold_answer",
    "gold_evidence_statement_details",
    "gold_paragraphs",
    "question",
    "retrieved_chunks",
    "statement_details",
}


def repo_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    return Path(out.strip())


def tracked_paths(root: Path) -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
    return [root / line for line in out.splitlines() if line]


def walk_paths(root: Path, roots: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in roots:
        base = root / item
        if base.is_file():
            paths.append(base)
        elif base.is_dir():
            paths.extend(p for p in base.rglob("*") if p.is_file())
    return paths


def is_scannable(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return False
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    if not rel.startswith(DEFAULT_SCAN_PREFIXES):
        return False
    return path.suffix.lower() in TEXT_SUFFIXES


def read_text_for_scan(path: Path) -> str:
    if path.suffix.lower() != ".ipynb":
        return path.read_text(encoding="utf-8", errors="ignore")

    try:
        notebook = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return ""

    # Code can contain column names such as "pemohon_nama"; scan only markdown
    # and outputs, because those are what readers see in a public notebook.
    visible_parts: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            visible_parts.extend(cell.get("source", []))
        for output in cell.get("outputs", []):
            text = output.get("text")
            if isinstance(text, list):
                visible_parts.extend(text)
            elif isinstance(text, str):
                visible_parts.append(text)
            data = output.get("data", {})
            for value in data.values():
                if isinstance(value, list):
                    visible_parts.extend(str(v) for v in value)
                elif isinstance(value, str):
                    visible_parts.append(value)
    return "\n".join(visible_parts)


def audit_jsonl_fields(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if path.suffix.lower() != ".jsonl":
        return counts

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_jsonl_record"] += 1
                continue
            for field in DISALLOWED_JSONL_FIELDS.intersection(row):
                counts[f"disallowed_jsonl_field:{field}"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Optional paths to scan instead of default roots.")
    parser.add_argument("--tracked", action="store_true", help="Scan git-tracked files.")
    parser.add_argument("--fail-on-signals", action="store_true")
    parser.add_argument("--fail-on-jsonl-free-text", action="store_true")
    parser.add_argument("--csv", dest="csv_path", help="Write a CSV report to this path.")
    args = parser.parse_args()

    root = repo_root()
    if args.tracked:
        candidates = tracked_paths(root)
    elif args.paths:
        candidates = walk_paths(root, args.paths)
    else:
        candidates = walk_paths(root, DEFAULT_SCAN_PREFIXES)

    files = sorted({p for p in candidates if p.exists() and is_scannable(p, root)})
    signal_counts: Counter[str] = Counter()
    files_with_signal: defaultdict[str, int] = defaultdict(int)
    jsonl_field_counts: Counter[str] = Counter()

    for path in files:
        text = read_text_for_scan(path)
        for name, pattern in PII_PATTERNS.items():
            count = len(pattern.findall(text))
            if count:
                signal_counts[name] += count
                files_with_signal[name] += 1
        jsonl_field_counts.update(audit_jsonl_fields(path))

    rows = []
    for name in sorted(PII_PATTERNS):
        rows.append(
            {
                "category": "pii_signal",
                "name": name,
                "matches": signal_counts[name],
                "files": files_with_signal[name],
            }
        )
    for name, count in sorted(jsonl_field_counts.items()):
        rows.append({"category": "jsonl_field", "name": name, "matches": count, "files": ""})

    writer = csv.DictWriter(sys.stdout, fieldnames=["category", "name", "matches", "files"])
    writer.writeheader()
    writer.writerows(rows)

    if args.csv_path:
        out = root / args.csv_path
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            file_writer = csv.DictWriter(fh, fieldnames=["category", "name", "matches", "files"])
            file_writer.writeheader()
            file_writer.writerows(rows)

    has_signal = any(signal_counts.values())
    has_disallowed_jsonl = any(jsonl_field_counts.values())
    if args.fail_on_signals and has_signal:
        return 2
    if args.fail_on_jsonl_free_text and has_disallowed_jsonl:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
