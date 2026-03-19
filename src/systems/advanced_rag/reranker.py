"""
Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Reranking flow:
  1. FAISS returns top-k chunks (e.g. k=10)
  2. Cross-encoder scores each (query, chunk) pair
  3. Top-m chunks returned (m ≤ k, default from config)
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.indexing.chunkers.fixed_size import Chunk
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str) -> CrossEncoder:
    """Load cross-encoder once and cache."""
    log.info(f"Loading cross-encoder: {model_name}")
    return CrossEncoder(model_name)


def rerank(
    query: str,
    chunks: list[Chunk],
    top_m: int | None = None,
    model_name: str | None = None,
) -> list[Chunk]:
    """
    Rerank retrieved chunks using a cross-encoder model.

    Args:
        query:      The user question (or rewritten query).
        chunks:     Candidates from initial dense retrieval.
        top_m:      Number of chunks to return after reranking.
        model_name: Cross-encoder model name.

    Returns:
        Top-m chunks sorted by cross-encoder score (descending).
    """
    if not chunks:
        return []

    top_m      = top_m      or cfg.rag.rerank_top_m
    model_name = model_name or cfg.models.reranker_model

    encoder = _get_cross_encoder(model_name)

    pairs  = [(query, c.text) for c in chunks]
    scores = encoder.predict(pairs)

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    result = [c for _, c in scored[:top_m]]

    log.debug(
        f"Reranked {len(chunks)} → {len(result)} chunks | "
        f"top score={scored[0][0]:.4f} bottom score={scored[-1][0]:.4f}"
    )
    return result
