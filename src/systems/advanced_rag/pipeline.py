"""
Advanced RAG pipeline — orchestrates all 4 optional components:
  1. Query Rewriting     (C-QR)
  2. Metadata Filtering  (C-MF)
  3. Hybrid Search       (C-HS)  — replaces dense-only if enabled
  4. Cross-Encoder Reranking (C-RR)

Each component is independently togglable via AblationFlags,
enabling the Section 3.3 ablation study with a single class.

Ablation conditions:
  Condition B   → all False (= Simple RAG baseline)
  C-QR          → use_query_rewrite=True only
  C-RR          → use_reranking=True only
  C-HS          → use_hybrid_search=True only
  C-MF          → use_metadata_filter=True only
  Full Advanced → all True
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import google.generativeai as genai
from openai import OpenAI

from src.data.section_extractor import VerdictSections
from src.indexing.chunkers.fixed_size import Chunk, FixedSizeChunker
from src.indexing.chunkers.section_boundary import SectionBoundaryChunker
from src.indexing.vector_store import VerdictIndex, VerdictIndexRegistry
from src.systems.advanced_rag.hybrid_search import HybridSearcher
from src.systems.advanced_rag.metadata_filter import MetadataFilter, apply_filter, infer_filter_from_question
from src.systems.advanced_rag.query_rewriter import rewrite_query
from src.systems.advanced_rag.reranker import rerank
from src.systems.base import QAResult, QASystem
from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.text_utils import build_prompt, join_chunks, SYSTEM_INSTRUCTION
from src.utils.token_counter import count_tokens, estimate_cost

log = get_logger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


@dataclass
class AblationFlags:
    """Toggle each Advanced RAG component independently."""
    use_query_rewrite:   bool = True
    use_metadata_filter: bool = True
    use_hybrid_search:   bool = True
    use_reranking:       bool = True

    @classmethod
    def simple_rag(cls) -> "AblationFlags":
        return cls(False, False, False, False)

    @classmethod
    def full_advanced(cls) -> "AblationFlags":
        return cls(True, True, True, True)

    @classmethod
    def query_rewrite_only(cls) -> "AblationFlags":
        return cls(use_query_rewrite=True, use_metadata_filter=False,
                   use_hybrid_search=False, use_reranking=False)

    @classmethod
    def reranking_only(cls) -> "AblationFlags":
        return cls(use_query_rewrite=False, use_metadata_filter=False,
                   use_hybrid_search=False, use_reranking=True)

    @classmethod
    def hybrid_only(cls) -> "AblationFlags":
        return cls(use_query_rewrite=False, use_metadata_filter=False,
                   use_hybrid_search=True, use_reranking=False)

    @classmethod
    def metadata_only(cls) -> "AblationFlags":
        return cls(use_query_rewrite=False, use_metadata_filter=True,
                   use_hybrid_search=False, use_reranking=False)

    @property
    def condition_suffix(self) -> str:
        parts = []
        if self.use_query_rewrite:   parts.append("QR")
        if self.use_metadata_filter: parts.append("MF")
        if self.use_hybrid_search:   parts.append("HS")
        if self.use_reranking:       parts.append("RR")
        return "_".join(parts) if parts else "simple_rag"


class AdvancedRAGSystem(QASystem):
    """
    Advanced RAG orchestration pipeline.

    Args:
        model:        LLM model name.
        flags:        AblationFlags controlling which components are active.
        chunk_size:   Token count per chunk.
        top_k:        Initial retrieval count (before reranking).
        chunking_strategy: 'fixed' | 'section'.
        registry:     VerdictIndexRegistry (shared across experiments).
    """

    def __init__(
        self,
        model: str | None = None,
        flags: AblationFlags | None = None,
        chunk_size: int | None = None,
        top_k: int | None = None,
        chunking_strategy: str = "fixed",
        registry: VerdictIndexRegistry | None = None,
        cleaned_dir: Path | None = None,
        sectioned_dir: Path | None = None,
    ):
        self.model             = model             or cfg.models.phase1_model
        self.flags             = flags             or AblationFlags.full_advanced()
        self.chunk_size        = chunk_size        or cfg.rag.chunk_size
        self.top_k             = top_k             or cfg.rag.top_k
        self.chunking_strategy = chunking_strategy
        self.registry          = registry          or VerdictIndexRegistry()
        self.cleaned_dir       = cleaned_dir       or cfg.paths.cleaned
        self.sectioned_dir     = sectioned_dir     or cfg.paths.sectioned

        # HybridSearcher cache (built per verdict on demand)
        self._hybrid_cache: dict[str, HybridSearcher] = {}

    @property
    def condition_name(self) -> str:
        return f"advanced_rag_{self.flags.condition_suffix}"

    # ── Index management ──────────────────────────────────────────────────────

    def _get_chunks(self, verdict_id: str) -> list[Chunk]:
        """Load or build chunks for a verdict."""
        if self.chunking_strategy == "section":
            sec_path = self.sectioned_dir / f"{verdict_id}.json"
            data     = json.loads(sec_path.read_text(encoding="utf-8"))
            sections = VerdictSections(**data)
            chunker  = SectionBoundaryChunker(max_chunk_chars=self.chunk_size * 4)
            return chunker.split(sections)
        else:
            txt  = (self.cleaned_dir / f"{verdict_id}.txt").read_text(encoding="utf-8")
            return FixedSizeChunker(chunk_size=self.chunk_size).split(txt, verdict_id)

    def _ensure_index(self, verdict_id: str) -> VerdictIndex:
        if not self.registry.exists(verdict_id):
            chunks = self._get_chunks(verdict_id)
            log.info(f"Building index for {verdict_id}: {len(chunks)} chunks")
            self.registry.build_and_save(verdict_id, chunks)
        return self.registry.get(verdict_id)

    def _ensure_hybrid(self, verdict_id: str) -> HybridSearcher:
        if verdict_id not in self._hybrid_cache:
            chunks = self._get_chunks(verdict_id)
            index  = self._ensure_index(verdict_id)
            self._hybrid_cache[verdict_id] = HybridSearcher(chunks, index)
        return self._hybrid_cache[verdict_id]

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        if self.model.startswith("gemini"):
            llm = genai.GenerativeModel(
                self.model,
                generation_config=genai.types.GenerationConfig(
                    temperature=cfg.models.temperature,
                    max_output_tokens=cfg.models.max_output_tokens,
                ),
            )
            return llm.generate_content(prompt).text.strip()
        response = _openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.models.temperature,
            max_tokens=cfg.models.max_output_tokens,
        )
        return response.choices[0].message.content.strip()

    # ── Main query ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        verdict_id: str,
        question_id: str = "",
        metadata_filter: MetadataFilter | None = None,
    ) -> QAResult:
        t0    = time.perf_counter()
        debug = {"components_used": []}

        # Step 1 — Query rewriting
        if self.flags.use_query_rewrite:
            queries = rewrite_query(question, model=self.model)
            debug["components_used"].append("query_rewrite")
            debug["rewritten_queries"] = queries
        else:
            queries = [question]

        # Step 2 — Metadata filter
        active_filter = None
        if self.flags.use_metadata_filter:
            active_filter = metadata_filter or infer_filter_from_question(question)
            debug["components_used"].append("metadata_filter")
            debug["filter_sections"] = active_filter.sections

        # Step 3 — Retrieval (hybrid or dense)
        all_candidates: list[Chunk] = []
        retrieval_t0 = time.perf_counter()

        for q in queries:
            if self.flags.use_hybrid_search:
                searcher = self._ensure_hybrid(verdict_id)
                if active_filter:
                    # Filter chunks before search
                    filtered_chunks = apply_filter(searcher.chunks, active_filter)
                    temp_index      = self.registry.get(verdict_id)
                    temp_searcher   = HybridSearcher(filtered_chunks, temp_index)
                    candidates      = temp_searcher.search(q, top_k=self.top_k)
                else:
                    candidates = searcher.search(q, top_k=self.top_k)
                if "hybrid_search" not in debug["components_used"]:
                    debug["components_used"].append("hybrid_search")
            else:
                index      = self._ensure_index(verdict_id)
                candidates = index.search(q, top_k=self.top_k)
                if active_filter:
                    candidates = apply_filter(candidates, active_filter)

            all_candidates.extend(candidates)

        # Deduplicate candidates by text
        seen_texts:    set[str]   = set()
        unique_chunks: list[Chunk] = []
        for c in all_candidates:
            if c.text not in seen_texts:
                seen_texts.add(c.text)
                unique_chunks.append(c)

        retrieval_latency = time.perf_counter() - retrieval_t0
        debug["n_candidates"] = len(unique_chunks)

        # Step 4 — Reranking
        if self.flags.use_reranking and len(unique_chunks) > cfg.rag.rerank_top_m:
            final_chunks = rerank(question, unique_chunks, top_m=cfg.rag.rerank_top_m)
            debug["components_used"].append("reranking")
        else:
            final_chunks = unique_chunks[:self.top_k]

        # Step 5 — Generation
        chunk_texts = [c.text for c in final_chunks]
        context     = join_chunks(chunk_texts)
        prompt      = build_prompt(SYSTEM_INSTRUCTION, context, question)

        input_tokens = count_tokens(prompt, self.model)
        cost_usd     = estimate_cost(input_tokens, self.model, cfg.model_pricing)

        t_gen     = time.perf_counter()
        answer    = self._call_llm(prompt)
        latency_s = time.perf_counter() - t0

        log.debug(
            f"[{self.condition_name}] {question_id} | "
            f"queries={len(queries)} chunks={len(final_chunks)} "
            f"tokens={input_tokens:,} cost=${cost_usd:.5f} "
            f"latency={latency_s:.2f}s (retrieval={retrieval_latency:.2f}s "
            f"gen={time.perf_counter()-t_gen:.2f}s)"
        )

        return QAResult(
            question_id=question_id,
            question=question,
            verdict_id=verdict_id,
            answer=answer,
            retrieved_chunks=chunk_texts,
            context_used=context,
            condition=self.condition_name,
            model=self.model,
            chunk_size=self.chunk_size,
            top_k=self.top_k,
            input_tokens=input_tokens,
            cost_usd=cost_usd,
            latency_s=latency_s,
            extra={
                **debug,
                "retrieval_latency_s": round(retrieval_latency, 4),
                "generation_latency_s": round(time.perf_counter() - t_gen, 4),
                "chunk_sections": [c.section for c in final_chunks],
            },
        )
