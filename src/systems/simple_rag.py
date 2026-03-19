"""
Simple RAG system: FAISS top-k dense retrieval → LLM generation.
No query rewriting, reranking, or hybrid search.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import google.generativeai as genai
from openai import OpenAI

from src.indexing.chunkers.fixed_size import FixedSizeChunker
from src.indexing.chunkers.section_boundary import SectionBoundaryChunker
from src.indexing.vector_store import VerdictIndexRegistry
from src.systems.base import QAResult, QASystem
from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.text_utils import build_prompt, join_chunks, SYSTEM_INSTRUCTION
from src.utils.token_counter import count_tokens, estimate_cost

log = get_logger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


class SimpleRAGSystem(QASystem):
    """
    Simple RAG: embed → retrieve top-k → generate.

    Args:
        model:            LLM model name.
        chunk_size:       Token count per chunk (256/512/1024).
        top_k:            Number of chunks to retrieve (3/5/10).
        chunking_strategy: 'fixed' | 'section' (for Section 8.4 experiment).
        registry:         Pre-built VerdictIndexRegistry (or build on demand).
    """

    def __init__(
        self,
        model: str | None = None,
        chunk_size: int | None = None,
        top_k: int | None = None,
        chunking_strategy: str = "fixed",
        registry: VerdictIndexRegistry | None = None,
        cleaned_dir: Path | None = None,
        sectioned_dir: Path | None = None,
    ):
        self.model             = model             or cfg.models.phase1_model
        self.chunk_size        = chunk_size        or cfg.rag.chunk_size
        self.top_k             = top_k             or cfg.rag.top_k
        self.chunking_strategy = chunking_strategy
        self.registry          = registry          or VerdictIndexRegistry()
        self.cleaned_dir       = cleaned_dir       or cfg.paths.cleaned
        self.sectioned_dir     = sectioned_dir     or cfg.paths.sectioned

    @property
    def condition_name(self) -> str:
        return f"simple_rag_cs{self.chunk_size}_k{self.top_k}_{self.chunking_strategy}"

    def _ensure_index(self, verdict_id: str) -> None:
        """Build and cache the FAISS index for a verdict if not already built."""
        if self.registry.exists(verdict_id):
            return

        if self.chunking_strategy == "section":
            from src.indexing.chunkers.section_boundary import SectionBoundaryChunker
            from src.data.section_extractor import VerdictSections
            import json
            sec_path = self.sectioned_dir / f"{verdict_id}.json"
            data = json.loads(sec_path.read_text(encoding="utf-8"))
            sections = VerdictSections(**data)
            chunker = SectionBoundaryChunker(max_chunk_chars=self.chunk_size * 4)
            chunks = chunker.split(sections)
        else:
            txt_path = self.cleaned_dir / f"{verdict_id}.txt"
            text = txt_path.read_text(encoding="utf-8")
            chunker = FixedSizeChunker(chunk_size=self.chunk_size)
            chunks = chunker.split(text, verdict_id=verdict_id)

        log.info(f"Building index for {verdict_id}: {len(chunks)} chunks")
        self.registry.build_and_save(verdict_id, chunks)

    def _call_llm(self, prompt: str) -> str:
        if self.model.startswith("gemini"):
            model = genai.GenerativeModel(
                self.model,
                generation_config=genai.types.GenerationConfig(
                    temperature=cfg.models.temperature,
                    max_output_tokens=cfg.models.max_output_tokens,
                ),
            )
            return model.generate_content(prompt).text.strip()
        else:
            response = _openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.models.temperature,
                max_tokens=cfg.models.max_output_tokens,
            )
            return response.choices[0].message.content.strip()

    def query(
        self,
        question: str,
        verdict_id: str,
        question_id: str = "",
    ) -> QAResult:
        t0 = time.perf_counter()

        # Retrieval
        self._ensure_index(verdict_id)
        index = self.registry.get(verdict_id)
        retrieved = index.search(question, top_k=self.top_k)
        retrieval_latency = time.perf_counter() - t0

        chunk_texts = [c.text for c in retrieved]
        context     = join_chunks(chunk_texts)
        prompt      = build_prompt(SYSTEM_INSTRUCTION, context, question)

        # Generation
        input_tokens = count_tokens(prompt, self.model)
        cost_usd     = estimate_cost(input_tokens, self.model, cfg.model_pricing)

        t_gen    = time.perf_counter()
        answer   = self._call_llm(prompt)
        latency_s = time.perf_counter() - t0

        log.debug(
            f"[{self.condition_name}] {question_id} | "
            f"chunks={len(retrieved)} tokens={input_tokens:,} "
            f"cost=${cost_usd:.5f} latency={latency_s:.2f}s "
            f"(retrieval={retrieval_latency:.2f}s gen={time.perf_counter()-t_gen:.2f}s)"
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
                "retrieval_latency_s": round(retrieval_latency, 4),
                "chunking_strategy": self.chunking_strategy,
                "chunk_sections": [c.section for c in retrieved],
            },
        )
