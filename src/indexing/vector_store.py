"""
FAISS-based vector store: build, save, load, and search.
One index per verdict (for per-document retrieval in RAG).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from src.indexing.chunkers.fixed_size import Chunk
from src.indexing.embedder import embed_chunks, embed_query
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


class VerdictIndex:
    """
    FAISS cosine similarity index for a single MK verdict.

    Stores:
      - FAISS flat index (L2 with normalized vectors = cosine similarity)
      - Chunk metadata (text, verdict_id, section, chunk_idx)
    """

    def __init__(self):
        self._index:  faiss.Index | None = None
        self._chunks: list[Chunk] = []
        self.dim:     int = 0

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Build index from pre-computed embeddings."""
        assert len(chunks) == len(embeddings), "chunks and embeddings length mismatch"
        self.dim = embeddings.shape[1]
        self._chunks = chunks

        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(normalized.astype(np.float32))
        log.debug(f"Built index: {len(chunks)} chunks, dim={self.dim}")

    def search(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """
        Retrieve top-k chunks for a query string.

        Args:
            query: Natural language question.
            top_k: Number of chunks to retrieve (default from config).

        Returns:
            List of Chunk objects, ranked by cosine similarity.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        top_k = top_k or cfg.rag.top_k
        q_emb = embed_query(query).reshape(1, -1)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)

        distances, indices = self._index.search(q_norm.astype(np.float32), top_k)
        return [self._chunks[i] for i in indices[0] if i >= 0]

    def save(self, path: Path) -> None:
        """Save index to disk (.faiss + .meta.pkl)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".meta.pkl"), "wb") as f:
            pickle.dump({"chunks": self._chunks, "dim": self.dim}, f)
        log.debug(f"Saved index → {path}")

    def load(self, path: Path) -> None:
        """Load index from disk."""
        self._index = faiss.read_index(str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self._chunks = meta["chunks"]
        self.dim     = meta["dim"]
        log.debug(f"Loaded index ← {path} ({len(self._chunks)} chunks)")


class VerdictIndexRegistry:
    """
    Registry of per-verdict FAISS indexes.
    Manages build / save / load for all 50 sample verdicts.
    """

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or cfg.paths.embedded
        self._cache: dict[str, VerdictIndex] = {}

    def _index_path(self, verdict_id: str) -> Path:
        return self.index_dir / verdict_id

    def build_and_save(self, verdict_id: str, chunks: list[Chunk]) -> VerdictIndex:
        embeddings = embed_chunks(chunks)
        idx = VerdictIndex()
        idx.build(chunks, embeddings)
        idx.save(self._index_path(verdict_id))
        self._cache[verdict_id] = idx
        return idx

    def get(self, verdict_id: str) -> VerdictIndex:
        if verdict_id in self._cache:
            return self._cache[verdict_id]
        idx = VerdictIndex()
        idx.load(self._index_path(verdict_id))
        self._cache[verdict_id] = idx
        return idx

    def exists(self, verdict_id: str) -> bool:
        return self._index_path(verdict_id).with_suffix(".faiss").exists()


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from src.indexing.chunkers.fixed_size import FixedSizeChunker

    parser = argparse.ArgumentParser(description="Build FAISS indexes for sampled verdicts.")
    parser.add_argument("--all", action="store_true", help="Index all 50 sample verdicts")
    parser.add_argument("--chunk-size", type=int, default=cfg.rag.chunk_size, help="Override chunk size")
    args = parser.parse_args()

    if args.all:
        if not cfg.paths.sample_50.exists():
            log.error(f"Sample file not found: {cfg.paths.sample_50}. Run sampler first.")
            exit(1)

        sample = pd.read_csv(cfg.paths.sample_50)
        registry = VerdictIndexRegistry()
        
        log.info(f"Building indexes for {len(sample)} verdicts (chunk_size={args.chunk_size})")
        
        for _, row in sample.iterrows():
            vid = row["file_id"]
            if registry.exists(vid):
                log.info(f"  Skipping {vid} (already indexed)")
                continue

            txt_path = cfg.paths.cleaned / f"{vid}.txt"
            if not txt_path.exists():
                log.warning(f"  Missing cleaned text: {vid}")
                continue

            text = txt_path.read_text(encoding="utf-8")
            chunks = FixedSizeChunker(chunk_size=args.chunk_size).split(text, verdict_id=vid)
            registry.build_and_save(vid, chunks)
            log.info(f"  Indexed {vid}: {len(chunks)} chunks")

        log.info("✓ FAISS index build complete.")
    else:
        parser.print_help()
