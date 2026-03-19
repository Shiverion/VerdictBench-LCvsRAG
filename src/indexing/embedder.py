"""
Google text-embedding-004 wrapper with batching and retry logic.
Fixed across all RAG conditions (controlled variable).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import google.generativeai as genai

from src.indexing.chunkers.fixed_size import Chunk
from src.utils.config import cfg
from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

_EMBED_BATCH_SIZE  = 100   # Google API batch limit
_RETRY_ATTEMPTS    = 3
_RETRY_DELAY_S     = 2.0


def embed_texts(
    texts: list[str],
    model: str | None = None,
    task_type: str = "retrieval_document",
) -> np.ndarray:
    """
    Embed a list of texts using Google's embedding API.

    Args:
        texts:     List of strings to embed.
        model:     Embedding model name (default from config).
        task_type: 'retrieval_document' for indexing, 'retrieval_query' for queries.

    Returns:
        ndarray of shape (len(texts), embedding_dim).
    """
    model = model or cfg.models.embedding_model
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]

        for attempt in range(_RETRY_ATTEMPTS):
            try:
                result = genai.embed_content(
                    model=model,
                    content=batch,
                    task_type=task_type,
                )
                all_embeddings.extend(result["embedding"])
                break
            except Exception as e:
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise RuntimeError(f"Embedding failed after {_RETRY_ATTEMPTS} attempts: {e}")
                log.warning(f"Embed attempt {attempt+1} failed: {e} — retrying in {_RETRY_DELAY_S}s")
                time.sleep(_RETRY_DELAY_S)

        log.debug(f"Embedded batch {i // _EMBED_BATCH_SIZE + 1} / {-(-len(texts) // _EMBED_BATCH_SIZE)}")

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(query: str, model: str | None = None) -> np.ndarray:
    """Embed a single query string."""
    model = model or cfg.models.embedding_model
    result = genai.embed_content(
        model=model,
        content=query,
        task_type="retrieval_query",
    )
    return np.array(result["embedding"], dtype=np.float32)


def embed_chunks(chunks: list[Chunk], model: str | None = None) -> np.ndarray:
    """Embed a list of Chunk objects. Returns (n_chunks, dim) array."""
    texts = [c.text for c in chunks]
    log.info(f"Embedding {len(texts)} chunks...")
    return embed_texts(texts, model=model)
