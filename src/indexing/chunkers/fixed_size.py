"""
Fixed-size chunker using recursive character splitting.
Chunk sizes evaluated: 256 / 512 / 1024 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import cfg


@dataclass
class Chunk:
    text:       str
    verdict_id: str
    chunk_idx:  int
    chunk_size: int     # configured chunk size (tokens)
    n_chars:    int     # actual char count of this chunk
    section:    str = "unknown"  # populated if section info available


class FixedSizeChunker:
    """
    Wraps LangChain's RecursiveCharacterTextSplitter.

    Token count approximation: 1 token ≈ 4 characters (Gemini convention).
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap_ratio: float | None = None):
        self.chunk_size          = chunk_size          or cfg.rag.chunk_size
        self.chunk_overlap_ratio = chunk_overlap_ratio or cfg.rag.chunk_overlap_ratio

        chars_per_chunk   = self.chunk_size * 4
        chars_overlap     = int(chars_per_chunk * self.chunk_overlap_ratio)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chars_per_chunk,
            chunk_overlap=chars_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, text: str, verdict_id: str = "") -> list[Chunk]:
        raw_chunks = self._splitter.split_text(text)
        return [
            Chunk(
                text=c,
                verdict_id=verdict_id,
                chunk_idx=i,
                chunk_size=self.chunk_size,
                n_chars=len(c),
            )
            for i, c in enumerate(raw_chunks)
        ]

    def split_with_sections(
        self,
        sections: dict[str, str],
        verdict_id: str = "",
    ) -> list[Chunk]:
        """
        Split per named section, preserving section label in each Chunk.
        sections: {"pertimbangan": "...", "duduk_perkara": "...", ...}
        """
        all_chunks: list[Chunk] = []
        global_idx = 0
        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue
            raw_chunks = self._splitter.split_text(section_text)
            for c in raw_chunks:
                all_chunks.append(
                    Chunk(
                        text=c,
                        verdict_id=verdict_id,
                        chunk_idx=global_idx,
                        chunk_size=self.chunk_size,
                        n_chars=len(c),
                        section=section_name,
                    )
                )
                global_idx += 1
        return all_chunks
