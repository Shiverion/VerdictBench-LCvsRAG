"""
Section-boundary chunker — aligns chunks to MK [X.X] paragraph boundaries.

This is a key differentiator for this study:
  - Fixed-size chunking may split a single paragraph across two chunks
  - Section-boundary chunking guarantees each [X.X] paragraph is a discrete unit

For very long pertimbangan paragraphs that exceed max_chunk_chars,
a secondary fixed-size split is applied within the paragraph.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data.section_extractor import VerdictSections, extract_sections
from src.indexing.chunkers.fixed_size import Chunk
from src.utils.config import cfg


class SectionBoundaryChunker:
    """
    Produces one Chunk per [X.X] paragraph.
    Oversized paragraphs are sub-split with a secondary splitter.
    """

    def __init__(self, max_chunk_chars: int | None = None):
        self.max_chunk_chars = max_chunk_chars or (cfg.rag.chunk_size * 4)
        self._secondary = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_chars,
            chunk_overlap=int(self.max_chunk_chars * cfg.rag.chunk_overlap_ratio),
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _paragraphs_to_chunks(
        self,
        paragraphs: list[dict],
        verdict_id: str,
        section_name: str,
        start_idx: int,
    ) -> tuple[list[Chunk], int]:
        """Convert a list of {marker, text} paragraphs into Chunks."""
        chunks: list[Chunk] = []
        idx = start_idx

        for para in paragraphs:
            text = para["text"]
            if len(text) <= self.max_chunk_chars:
                chunks.append(Chunk(
                    text=text,
                    verdict_id=verdict_id,
                    chunk_idx=idx,
                    chunk_size=cfg.rag.chunk_size,
                    n_chars=len(text),
                    section=section_name,
                ))
                idx += 1
            else:
                # Oversized paragraph: secondary split
                sub_chunks = self._secondary.split_text(text)
                for sub in sub_chunks:
                    chunks.append(Chunk(
                        text=sub,
                        verdict_id=verdict_id,
                        chunk_idx=idx,
                        chunk_size=cfg.rag.chunk_size,
                        n_chars=len(sub),
                        section=f"{section_name}_sub",
                    ))
                    idx += 1

        return chunks, idx

    def split(self, sections: VerdictSections) -> list[Chunk]:
        """
        Split a VerdictSections into Chunks aligned to paragraph boundaries.

        Args:
            sections: Parsed VerdictSections from section_extractor.

        Returns:
            List of Chunk objects.
        """
        all_chunks: list[Chunk] = []
        idx = 0

        for section_name in ["pembukaan", "duduk_perkara", "pertimbangan", "konklusi"]:
            paragraphs = getattr(sections, section_name, [])
            new_chunks, idx = self._paragraphs_to_chunks(
                paragraphs, sections.verdict_id, section_name, idx
            )
            all_chunks.extend(new_chunks)

        # AMAR as a single chunk
        if sections.amar_putusan:
            amar_text = sections.amar_putusan
            if len(amar_text) <= self.max_chunk_chars:
                all_chunks.append(Chunk(
                    text=amar_text,
                    verdict_id=sections.verdict_id,
                    chunk_idx=idx,
                    chunk_size=cfg.rag.chunk_size,
                    n_chars=len(amar_text),
                    section="amar_putusan",
                ))
            else:
                for sub in self._secondary.split_text(amar_text):
                    all_chunks.append(Chunk(
                        text=sub,
                        verdict_id=sections.verdict_id,
                        chunk_idx=idx,
                        chunk_size=cfg.rag.chunk_size,
                        n_chars=len(sub),
                        section="amar_putusan_sub",
                    ))
                    idx += 1

        return all_chunks

    def split_from_file(self, sectioned_json: Path) -> list[Chunk]:
        """Load a sectioned JSON file and split."""
        data = json.loads(sectioned_json.read_text(encoding="utf-8"))
        sections = VerdictSections(**data)
        return self.split(sections)
