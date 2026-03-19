"""
Corpus audit — sweeps all 887 raw verdicts and produces:
  - data/metadata/corpus_stats.csv        (one row per verdict)
  - data/metadata/amar_mismatches.csv     (JSON amar ≠ text amar)

Run:  python -m src.data.audit
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Regex patterns for text-derived fields ────────────────────────────────────
AMAR_PATTERN = re.compile(
    r"(Mengabulkan|Menolak|Tidak Dapat Diterima|menyatakan tidak berwenang|"
    r"Permohonan tidak dapat diterima)",
    re.IGNORECASE,
)

PANEL_KETUA_PATTERN = re.compile(
    r"(?:selaku\s+)?Ketua\s+merangkap\s+Anggota[,\s]+([A-Z][a-zA-Z\s\.]+?)(?:,|\n)",
)

TANGGAL_PUTUSAN_PATTERN = re.compile(
    r"(?:hari|tanggal)\s+\w+\s*,\s*tanggal\s+([\w\s]+),\s*bulan\s+([\w]+),\s*tahun\s+([\w\s]+)",
    re.IGNORECASE,
)

SECTION_MARKER = re.compile(r"\[(\d+)\.(\d+)(?:\.\d+)?\]")


def _extract_text_amar(text: str) -> str | None:
    m = AMAR_PATTERN.search(text)
    return m.group(0).title() if m else None


def _count_pages(text: str) -> int:
    """Estimate page count from standalone page-number lines."""
    numbers = re.findall(r"^\s*(\d{1,3})\s*$", text, re.MULTILINE)
    if numbers:
        return max(int(n) for n in numbers)
    # Fallback: ~2000 chars per page
    return max(1, len(text) // 2000)


def _count_section_markers(text: str) -> dict[str, int]:
    markers = SECTION_MARKER.findall(text)
    counts: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0}
    for major, _ in markers:
        if major in counts:
            counts[major] += 1
    return counts


def audit_verdict(txt_path: Path, json_path: Path) -> dict:
    """Produce a single-verdict audit record."""
    text = txt_path.read_text(encoding="utf-8", errors="replace")

    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    metadata   = meta.get("metadata", {})
    amar_json  = meta.get("amar_putusan", {}).get("status")
    amar_text  = _extract_text_amar(text)
    markers    = _count_section_markers(text)

    return {
        # Identifiers
        "file_id":           txt_path.stem,
        "nomor_putusan":     metadata.get("nomor_putusan"),
        "jenis_perkara":     metadata.get("jenis_perkara"),
        # Dates from JSON
        "tanggal_registrasi": metadata.get("tanggal_registrasi"),
        "tanggal_putusan":   metadata.get("tanggal_putusan"),
        # Size
        "n_chars":           len(text),
        "n_lines":           text.count("\n"),
        "est_pages":         _count_pages(text),
        # Section structure
        "n_pembukaan":       markers["1"],
        "n_duduk_perkara":   markers["2"],
        "n_pertimbangan":    markers["3"],
        "n_konklusi":        markers["4"],
        "has_amar_section":  bool(re.search(r"AMAR PUTUSAN", text, re.IGNORECASE)),
        # Amar reconciliation
        "amar_json":         amar_json,
        "amar_text":         amar_text,
        "amar_mismatch":  (
            amar_json is not None
            and amar_text is not None
            and amar_json.strip().lower() not in amar_text.strip().lower()
        ),
        # Metadata completeness
        "has_panel_ketua":   bool(meta.get("panel_hakim", {}).get("ketua")),
        "has_tanggal_putusan": bool(metadata.get("tanggal_putusan")),
        "has_pemohon_nama":  bool(meta.get("pemohon", {}).get("nama")),
        "n_norma_diuji":     len(meta.get("norma_diuji", [])),
        # JSON file exists
        "json_exists":       json_path.exists(),
    }


def run_audit(
    raw_txt_dir: Path | None = None,
    raw_json_dir: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Audit all verdict pairs and write CSV outputs.

    Returns:
        DataFrame of corpus_stats.
    """
    raw_txt_dir  = raw_txt_dir  or cfg.paths.raw_txt
    raw_json_dir = raw_json_dir or cfg.paths.raw_json
    out_dir      = out_dir      or cfg.paths.metadata_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_txt_dir.glob("*.txt"))
    log.info(f"Auditing {len(txt_files)} verdict files...")

    records = []
    for txt_path in tqdm(txt_files, desc="Auditing"):
        json_path = raw_json_dir / (txt_path.stem + ".json")
        try:
            record = audit_verdict(txt_path, json_path)
        except Exception as e:
            log.error(f"Failed on {txt_path.name}: {e}")
            record = {"file_id": txt_path.stem, "error": str(e)}
        records.append(record)

    df = pd.DataFrame(records)

    # Write corpus_stats.csv
    stats_path = out_dir / "corpus_stats.csv"
    df.to_csv(stats_path, index=False)
    log.info(f"Corpus stats → {stats_path} ({len(df)} rows)")

    # Write amar_mismatches.csv
    mismatches = df[df.get("amar_mismatch", False) == True]
    mismatch_path = out_dir / "amar_mismatches.csv"
    mismatches.to_csv(mismatch_path, index=False)
    log.info(f"Amar mismatches → {mismatch_path} ({len(mismatches)} rows)")

    # Summary
    log.info("\n=== CORPUS SUMMARY ===")
    log.info(f"Total verdicts:        {len(df)}")
    if "jenis_perkara" in df.columns:
        log.info(f"By type:\n{df.jenis_perkara.value_counts().to_string()}")
    if "amar_mismatch" in df.columns:
        log.info(f"Amar mismatches:       {df.amar_mismatch.sum()}")
    if "n_chars" in df.columns:
        log.info(f"Char count — min:{df.n_chars.min():,}  "
                 f"median:{int(df.n_chars.median()):,}  "
                 f"max:{df.n_chars.max():,}")

    return df


if __name__ == "__main__":
    run_audit()
