"""
Text normalization for MK verdict .txt files.

Problems to fix (observed in 1_PUU-XIX_2021.txt):
  1. Windows CRLF line endings (\r\n)
  2. PDF hyphen word-breaks: "berlak-\nunya" → "berlakunya"
  3. Standalone page number lines (e.g. a line containing just "7")
  4. Multiple consecutive blank lines → max 2
  5. Leading/trailing whitespace per line
"""

from __future__ import annotations

import re
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)


def clean_text(text: str) -> str:
    """Full normalization pipeline for a single verdict text."""

    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Rejoin hyphenated word-breaks from PDF rendering
    #    Pattern: word-\nnextword  →  wordnextword
    text = re.sub(r"-\n(\S)", r"\1", text)

    # 3. Remove standalone page number lines
    #    Lines that contain only digits (1–3 digits), optionally surrounded by whitespace
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # 4. Strip trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # 5. Collapse 3+ consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Strip leading/trailing whitespace from the whole document
    return text.strip()


def clean_file(src: Path, dst: Path) -> None:
    """Clean a single .txt file and write to dst."""
    raw = src.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")
    log.debug(f"Cleaned: {src.name} ({len(raw):,} → {len(cleaned):,} chars)")


def clean_corpus(raw_dir: Path, out_dir: Path) -> int:
    """
    Clean all .txt files in raw_dir and write to out_dir.

    Returns:
        Number of files processed.
    """
    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        log.warning(f"No .txt files found in {raw_dir}")
        return 0

    log.info(f"Cleaning {len(txt_files)} verdict files → {out_dir}")
    for src in txt_files:
        dst = out_dir / src.name
        clean_file(src, dst)

    log.info("Cleaning complete.")
    return len(txt_files)


if __name__ == "__main__":
    from src.utils.config import cfg
    clean_corpus(cfg.paths.raw_txt, cfg.paths.cleaned)
