"""
Merge JSON metadata + text-derived fields into verdicts_metadata.csv.
Fills null fields in original JSON where possible by parsing the verdict text.

Fields enriched:
  - tanggal_putusan   (from amar section date)
  - jumlah_halaman    (from page count estimation)
  - panel_hakim       (ketua + anggota from amar section)
  - amar_putusan.teks_asli (from AMAR PUTUSAN section)
  - stratum           (short / medium / long based on n_chars)
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


# Month name → number (Indonesian)
BULAN_MAP = {
    "januari": "01", "februari": "02", "maret": "03", "april": "04",
    "mei": "05", "juni": "06", "juli": "07", "agustus": "08",
    "september": "09", "oktober": "10", "november": "11", "desember": "12",
}

ANGKA_MAP = {
    "satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5",
    "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9",
    "sepuluh": "10", "sebelas": "11", "dua belas": "12", "tiga belas": "13",
    "empat belas": "14", "lima belas": "15", "enam belas": "16",
    "tujuh belas": "17", "delapan belas": "18", "sembilan belas": "19",
    "dua puluh": "20", "dua puluh satu": "21", "dua puluh dua": "22",
    "dua puluh tiga": "23", "dua puluh empat": "24", "dua puluh lima": "25",
    "dua puluh enam": "26", "dua puluh tujuh": "27", "dua puluh delapan": "28",
    "dua puluh sembilan": "29", "tiga puluh": "30", "tiga puluh satu": "31",
    # Thousands (years)
    "dua ribu dua puluh satu": "2021", "dua ribu dua puluh dua": "2022",
    "dua ribu dua puluh tiga": "2023", "dua ribu dua puluh empat": "2024",
    "dua ribu dua puluh lima": "2025",
}


def _words_to_number(words: str) -> str:
    w = words.strip().lower()
    return ANGKA_MAP.get(w, w)


def _extract_tanggal_putusan(text: str) -> str | None:
    """Extract verdict date from the deliberation section text."""
    pat = re.compile(
        r"pada\s+hari\s+\w+\s*,\s*tanggal\s+([a-zA-Z\s]+),\s*bulan\s+([a-zA-Z]+),"
        r"\s*tahun\s+([a-zA-Z\s]+)",
        re.IGNORECASE,
    )
    m = pat.search(text)
    if not m:
        return None
    day   = _words_to_number(m.group(1))
    month = BULAN_MAP.get(m.group(2).strip().lower())
    year  = _words_to_number(m.group(3))
    if day and month and year and year.isdigit():
        return f"{year}-{month}-{day.zfill(2)}"
    return None


def _extract_panel(text: str) -> dict:
    """Extract panel hakim from verdict closing section."""
    panel = {"ketua": None, "anggota": []}

    ketua_m = re.search(
        r"([A-Z][a-zA-Z\s\.]+?),?\s+selaku\s+Ketua\s+merangkap\s+Anggota",
        text,
    )
    if ketua_m:
        panel["ketua"] = ketua_m.group(1).strip()

    # Extract all named Hakim Konstitusi
    anggota_m = re.findall(
        r"Hakim\s+Konstitusi\s+(?:yaitu\s+)?([A-Z][a-zA-Z\s\.,]+?)(?:\s+dan\s+|\s*,\s*|\s+masing)",
        text,
    )
    if anggota_m:
        panel["anggota"] = [a.strip().rstrip(",") for a in anggota_m if a.strip()]

    return panel


def _extract_amar_teks(text: str) -> str | None:
    m = re.search(
        r"Mengadili\s*:(.*?)(?=\nDemikian|\nKETUA|\Z)",
        text, re.DOTALL | re.IGNORECASE,
    )
    return m.group(0).strip() if m else None


def _assign_stratum(n_chars: int) -> str:
    if n_chars < cfg.sampling.short_max_chars:
        return "short"
    elif n_chars >= cfg.sampling.long_min_chars:
        return "long"
    return "medium"


def enrich_verdict(txt_path: Path, json_path: Path) -> dict:
    """Produce an enriched metadata record for a single verdict."""
    text = txt_path.read_text(encoding="utf-8", errors="replace")

    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    metadata = meta.get("metadata", {})
    panel    = meta.get("panel_hakim", {})
    amar     = meta.get("amar_putusan", {})

    # Text-derived enrichments
    tanggal_putusan_derived = _extract_tanggal_putusan(text)
    panel_derived           = _extract_panel(text)
    amar_teks_derived       = _extract_amar_teks(text)

    return {
        "file_id":              txt_path.stem,
        "nomor_putusan":        metadata.get("nomor_putusan"),
        "jenis_perkara":        metadata.get("jenis_perkara"),
        "undang_undang_teruji": metadata.get("undang_undang_teruji"),
        "tanggal_registrasi":   metadata.get("tanggal_registrasi"),
        # Prefer JSON if not null, else use text-derived
        "tanggal_putusan":      metadata.get("tanggal_putusan") or tanggal_putusan_derived,
        "tanggal_ucapan":       metadata.get("tanggal_ucapan"),
        "pemohon_nama":         meta.get("pemohon", {}).get("nama"),
        "pemohon_kategori":     meta.get("pemohon", {}).get("kategori_pemohon"),
        "n_norma_diuji":        len(meta.get("norma_diuji", [])),
        "amar_status":          amar.get("status") or amar_teks_derived,
        "amar_teks_asli":       amar.get("teks_asli") or amar_teks_derived,
        "panel_ketua":          panel.get("ketua") or panel_derived.get("ketua"),
        "panel_anggota":        json.dumps(panel.get("anggota") or panel_derived.get("anggota", []),
                                           ensure_ascii=False),
        # Size
        "n_chars":              len(text),
        "est_pages":            max(1, len(text) // 2000),
        "stratum":              _assign_stratum(len(text)),
    }


def enrich_corpus(
    raw_txt_dir: Path | None = None,
    raw_json_dir: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Enrich all verdict metadata and write verdicts_metadata.csv."""
    raw_txt_dir  = raw_txt_dir  or cfg.paths.raw_txt
    raw_json_dir = raw_json_dir or cfg.paths.raw_json
    out_dir      = out_dir      or cfg.paths.metadata_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_txt_dir.glob("*.txt"))
    log.info(f"Enriching metadata for {len(txt_files)} verdicts...")

    records = []
    for txt_path in tqdm(txt_files, desc="Enriching"):
        json_path = raw_json_dir / (txt_path.stem + ".json")
        try:
            records.append(enrich_verdict(txt_path, json_path))
        except Exception as e:
            log.error(f"Error enriching {txt_path.name}: {e}")

    df = pd.DataFrame(records)
    out_path = out_dir / "verdicts_metadata.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Enriched metadata → {out_path}")

    log.info(f"\nStratum distribution:\n{df.stratum.value_counts().to_string()}")
    log.info(f"Jenis perkara:\n{df.jenis_perkara.value_counts().to_string()}")

    return df


if __name__ == "__main__":
    enrich_corpus()
