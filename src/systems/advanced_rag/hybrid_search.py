"""
Hybrid search: dense (FAISS) + sparse (BM25) via Reciprocal Rank Fusion (RRF).

RRF formula:  score(d) = Σ  1 / (k + rank_i(d))
  where k=60 is the standard RRF constant.

Why this matters for MK verdicts:
  - Dense retrieval captures semantic similarity
  - BM25 captures exact legal terms: pasal numbers, case IDs, Latin maxims
  - RRF fusion improves recall for queries mixing both signals
"""

from __future__ import annotations

from collections import defaultdict

from rank_bm25 import BM25Okapi

from src.indexing.chunkers.fixed_size import Chunk
from src.indexing.vector_store import VerdictIndex
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

RRF_K = 60   # standard constant


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class HybridSearcher:
    """
    Combines FAISS dense index and BM25 sparse index for a single verdict.

    Usage:
        searcher = HybridSearcher(chunks, faiss_index)
        results  = searcher.search(query, top_k=10)
    """

    def __init__(self, chunks: list[Chunk], dense_index: VerdictIndex):
        self.chunks      = chunks
        self.dense_index = dense_index

        # Build BM25 from chunk texts
        tokenized = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        log.debug(f"HybridSearcher initialized with {len(chunks)} chunks")

    def search(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """
        Retrieve top-k chunks using RRF fusion of dense + sparse results.

        Args:
            query: Natural language question.
            top_k: Number of final chunks to return.

        Returns:
            Chunks ranked by RRF score.
        """
        top_k = top_k or cfg.rag.top_k

        # --- Dense retrieval (FAISS) ---
        # Retrieve 2×top_k candidates from each to give RRF enough to work with
        n_candidates = min(top_k * 2, len(self.chunks))
        dense_results = self.dense_index.search(query, top_k=n_candidates)

        # --- Sparse retrieval (BM25) ---
        query_tokens = _tokenize(query)
        bm25_scores  = self.bm25.get_scores(query_tokens)
        bm25_ranked  = sorted(
            range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True
        )[:n_candidates]
        sparse_results = [self.chunks[i] for i in bm25_ranked]

        # --- RRF fusion ---
        rrf_scores: dict[int, float] = defaultdict(float)

        # Map chunk text → index for deduplication
        chunk_by_idx = {i: c for i, c in enumerate(self.chunks)}
        text_to_idx  = {c.text: i for i, c in enumerate(self.chunks)}

        for rank, chunk in enumerate(dense_results):
            idx = text_to_idx.get(chunk.text, -1)
            if idx >= 0:
                rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

        for rank, chunk in enumerate(sparse_results):
            idx = text_to_idx.get(chunk.text, -1)
            if idx >= 0:
                rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

        # Sort by RRF score and return top-k
        ranked_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        result = [chunk_by_idx[i] for i in ranked_indices[:top_k]]

        log.debug(
            f"HybridSearch: dense={len(dense_results)} sparse={len(sparse_results)} "
            f"→ fused={len(result)}"
        )
        return result
