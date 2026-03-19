"""Shared text helpers used across the pipeline."""

from __future__ import annotations

import re


def truncate_text(text: str, max_chars: int, from_end: bool = False) -> str:
    """Truncate text to max_chars, optionally from the end."""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:] if from_end else text[:max_chars]


def join_chunks(chunks: list[str], separator: str = "\n\n---\n\n") -> str:
    """Join retrieved chunks with a visible separator for the LLM prompt."""
    return separator.join(c.strip() for c in chunks if c.strip())


def build_prompt(
    system_instruction: str,
    context: str,
    question: str,
    answer_instruction: str = (
        "Jawab pertanyaan di atas berdasarkan konteks yang diberikan. "
        "Jawablah dalam Bahasa Indonesia. "
        "Jika jawabannya tidak ada dalam konteks, katakan 'Informasi tidak tersedia dalam dokumen.'"
    ),
) -> str:
    """
    Canonical prompt template — identical across all system conditions.
    This is the single template referenced in the experimental design.
    """
    return (
        f"{system_instruction}\n\n"
        f"=== KONTEKS DOKUMEN ===\n{context}\n"
        f"=== AKHIR KONTEKS ===\n\n"
        f"Pertanyaan: {question}\n\n"
        f"{answer_instruction}"
    )


SYSTEM_INSTRUCTION = (
    "Anda adalah asisten hukum yang membantu menjawab pertanyaan berdasarkan "
    "putusan Mahkamah Konstitusi Republik Indonesia. "
    "Jawablah hanya berdasarkan informasi yang tersedia dalam konteks yang diberikan. "
    "Jangan menambahkan informasi di luar konteks."
)
