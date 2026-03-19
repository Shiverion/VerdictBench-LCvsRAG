"""
MK verdict section boundary extractor.

MK verdicts follow a consistent [X.X] paragraph numbering system:
  [1.X]         → Pembukaan / identitas perkara
  [2.X]         → Duduk Perkara (petitioner arguments + evidence)
  [3.X] / [3.X.X] → Pertimbangan Mahkamah (core legal reasoning)
  [4.X]         → Konklusi
  5. AMAR PUTUSAN → Operative ruling (not bracket-numbered)

This extractor produces a structured dict per verdict,
enabling section-boundary chunking and metadata-filtered retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Section label mapping ──────────────────────────────────────────────────────
SECTION_GROUPS = {
    "pembukaan":         re.compile(r"^\[1\.\d+\]"),
    "duduk_perkara":     re.compile(r"^\[2\.\d+\]"),
    "pertimbangan":      re.compile(r"^\[3\.\d+(?:\.\d+)?\]"),
    "konklusi":          re.compile(r"^\[4\.\d+\]"),
}

AMAR_PATTERN = re.compile(
    r"(5\.\s*AMAR PUTUSAN.*?)(?=\nKETUA|\nPANITERA|\Z)",
    re.DOTALL | re.IGNORECASE,
)

PARAGRAPH_PATTERN = re.compile(r"(\[\d+\.\d+(?:\.\d+)?\])")


@dataclass
class VerdictSections:
    verdict_id:      str
    pembukaan:       list[dict] = field(default_factory=list)   # [{marker, text}]
    duduk_perkara:   list[dict] = field(default_factory=list)
    pertimbangan:    list[dict] = field(default_factory=list)
    konklusi:        list[dict] = field(default_factory=list)
    amar_putusan:    Optional[str] = None
    full_text:       str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def section_text(self, section: str) -> str:
        """Return all paragraphs of a section concatenated."""
        paragraphs = getattr(self, section, [])
        if isinstance(paragraphs, list):
            return "\n\n".join(p["text"] for p in paragraphs)
        return paragraphs or ""

    def all_sections_text(self, exclude: list[str] | None = None) -> str:
        """Return full text reconstructed from sections (excluding specified)."""
        exclude = exclude or []
        parts = []
        for sec in ["pembukaan", "duduk_perkara", "pertimbangan", "konklusi"]:
            if sec not in exclude:
                parts.append(self.section_text(sec))
        if "amar_putusan" not in exclude and self.amar_putusan:
            parts.append(self.amar_putusan)
        return "\n\n".join(p for p in parts if p)


def extract_sections(text: str, verdict_id: str = "") -> VerdictSections:
    """
    Parse a cleaned MK verdict text into structured sections.

    Args:
        text:       Cleaned verdict text.
        verdict_id: Identifier string for logging.

    Returns:
        VerdictSections dataclass.
    """
    result = VerdictSections(verdict_id=verdict_id, full_text=text)

    # Split on [X.X] markers while keeping the markers
    parts = PARAGRAPH_PATTERN.split(text)

    # parts alternates: [pre_text, marker, content, marker, content, ...]
    i = 1
    while i < len(parts) - 1:
        marker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        paragraph = {"marker": marker, "text": f"{marker} {content}".strip()}

        for section_name, pattern in SECTION_GROUPS.items():
            if pattern.match(marker):
                getattr(result, section_name).append(paragraph)
                break

        i += 2

    # Extract AMAR PUTUSAN (not bracket-numbered)
    amar_match = AMAR_PATTERN.search(text)
    if amar_match:
        result.amar_putusan = amar_match.group(1).strip()

    if not result.pertimbangan:
        log.warning(f"{verdict_id}: No pertimbangan sections found — check document structure.")

    return result


def extract_and_save(cleaned_txt: Path, out_dir: Path) -> VerdictSections:
    """Extract sections from a cleaned .txt file and save as JSON."""
    text = cleaned_txt.read_text(encoding="utf-8")
    sections = extract_sections(text, verdict_id=cleaned_txt.stem)

    out_path = out_dir / f"{cleaned_txt.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(sections.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return sections


def extract_corpus(cleaned_dir: Path, out_dir: Path) -> int:
    """Extract sections for all cleaned verdicts. Returns file count."""
    files = sorted(cleaned_dir.glob("*.txt"))
    log.info(f"Extracting sections from {len(files)} verdicts → {out_dir}")
    for f in files:
        extract_and_save(f, out_dir)
    log.info("Section extraction complete.")
    return len(files)


if __name__ == "__main__":
    from src.utils.config import cfg
    extract_corpus(cfg.paths.cleaned, cfg.paths.sectioned)
