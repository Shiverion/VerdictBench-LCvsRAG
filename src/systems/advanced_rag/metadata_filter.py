"""
Metadata-based pre-filtering for Advanced RAG.

Filters the chunk pool before embedding search, using structured metadata
extracted from the verdict JSON and section labels.

Filter dimensions:
  1. Section filter    — only retrieve from specific sections (e.g. pertimbangan only)
  2. Case type filter  — within a multi-verdict index, limit to matching jenis_perkara
  3. Date filter       — limit to verdicts within a date range
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.indexing.chunkers.fixed_size import Chunk
from src.utils.logger import get_logger

log = get_logger(__name__)

# MK section names from section_extractor.py
VALID_SECTIONS = frozenset({
    "pembukaan",
    "duduk_perkara",
    "pertimbangan",
    "pertimbangan_sub",
    "konklusi",
    "amar_putusan",
    "amar_putusan_sub",
    "unknown",
})


@dataclass
class MetadataFilter:
    """
    Filter specification for chunk pre-filtering.

    All fields are optional — only specified fields are applied.
    Multiple fields are ANDed together.

    Attributes:
        sections:        If set, only chunks from these section names are kept.
        verdict_ids:     If set, only chunks from these verdict IDs are kept.
        min_chars:       Minimum chunk character length (removes stub chunks).
        exclude_sections: Sections to always exclude.
    """
    sections:         Optional[list[str]] = None
    verdict_ids:      Optional[list[str]] = None
    min_chars:        int = 50
    exclude_sections: list[str] = None

    def __post_init__(self):
        if self.exclude_sections is None:
            self.exclude_sections = []
        # Validate section names
        if self.sections:
            invalid = set(self.sections) - VALID_SECTIONS
            if invalid:
                log.warning(f"Unknown section names in filter: {invalid}")


def apply_filter(chunks: list[Chunk], f: MetadataFilter) -> list[Chunk]:
    """
    Apply a MetadataFilter to a list of chunks.

    Args:
        chunks: Full chunk pool for a verdict.
        f:      MetadataFilter specification.

    Returns:
        Filtered subset of chunks.
    """
    result = chunks

    # Section whitelist
    if f.sections:
        result = [c for c in result if c.section in f.sections]

    # Section blacklist
    if f.exclude_sections:
        result = [c for c in result if c.section not in f.exclude_sections]

    # Verdict ID filter (for multi-verdict indexes)
    if f.verdict_ids:
        result = [c for c in result if c.verdict_id in f.verdict_ids]

    # Minimum chunk length
    result = [c for c in result if c.n_chars >= f.min_chars]

    log.debug(f"MetadataFilter: {len(chunks)} → {len(result)} chunks")
    return result


def infer_filter_from_question(question: str) -> MetadataFilter:
    """
    Heuristic: infer which sections are most relevant based on question keywords.

    This is used when no explicit filter is provided — the system tries to
    narrow the search space based on what the question is likely asking about.

    Args:
        question: Natural language question in Indonesian.

    Returns:
        A MetadataFilter with inferred section constraints.
    """
    q = question.lower()

    # Amar/ruling questions → prioritize konklusi + amar
    if any(kw in q for kw in ["amar", "putusan", "mengadili", "menolak", "mengabulkan",
                                "dikabulkan", "ditolak", "konklusi"]):
        return MetadataFilter(sections=["konklusi", "amar_putusan"])

    # Reasoning questions → prioritize pertimbangan
    if any(kw in q for kw in ["pertimbangan", "alasan", "mengapa", "kenapa",
                                "dasar hukum", "mempertimbangkan", "mahkamah berpendapat"]):
        return MetadataFilter(sections=["pertimbangan", "pertimbangan_sub"])

    # Petitioner/background → duduk perkara
    if any(kw in q for kw in ["pemohon", "permohonan", "dalil", "bukti", "saksi",
                                "ahli", "keterangan", "mengajukan"]):
        return MetadataFilter(sections=["duduk_perkara", "pembukaan"])

    # Default: search all meaningful sections (exclude opening boilerplate)
    return MetadataFilter(
        exclude_sections=["pembukaan"],
        min_chars=100,
    )
