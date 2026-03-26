"""
Long Context (LC) system.
Injects full verdict text into the prompt — no retrieval.

Variants:
  - LC Full:     entire cleaned text (up to model context limit)
  - LC Windowed: first N + last N chars for models with shorter contexts (GPT-4o)
"""

from __future__ import annotations

import os
import time
import time
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI

from src.systems.base import QAResult, QASystem
from src.utils.config import cfg
from src.utils.logger import get_logger
from src.utils.text_utils import build_prompt, SYSTEM_INSTRUCTION, truncate_text
from src.utils.token_counter import count_tokens, estimate_cost

log = get_logger(__name__)

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))
_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# GPT-4o safe context limit (tokens → approx chars)
GPT4O_MAX_CONTEXT_CHARS = 100_000 * 4   # ~100k tokens
GPT4O_WINDOW_HALF       = 50_000 * 4    # take first + last 50k tokens each


class LongContextSystem(QASystem):
    """
    Full-document injection QA system.

    Args:
        model:        LLM model name.
        cleaned_dir:  Directory containing cleaned .txt verdict files.
        windowed:     If True, apply truncation window for models with limited context.
    """

    def __init__(
        self,
        model: str | None = None,
        cleaned_dir: Path | None = None,
        windowed: bool = False,
    ):
        self.model       = model or cfg.models.phase1_model
        self.cleaned_dir = cleaned_dir or cfg.paths.cleaned
        self.windowed    = windowed

    @property
    def condition_name(self) -> str:
        return "lc_windowed" if self.windowed else "lc"

    def _load_text(self, verdict_id: str) -> str:
        path = self.cleaned_dir / f"{verdict_id}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Cleaned verdict not found: {path}")
        text = path.read_text(encoding="utf-8")

        if self.windowed and len(text) > GPT4O_MAX_CONTEXT_CHARS:
            first_half = text[:GPT4O_WINDOW_HALF]
            last_half  = text[-GPT4O_WINDOW_HALF:]
            text = (
                first_half
                + "\n\n[... TENGAH DOKUMEN DIPOTONG UNTUK KETERBATASAN KONTEKS ...]\n\n"
                + last_half
            )
            log.debug(f"{verdict_id}: windowed to {len(text):,} chars")

        return text

    def _call_gemini(self, prompt: str) -> str:
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=cfg.models.temperature,
                max_output_tokens=cfg.models.max_output_tokens,
            ),
        )
        return response.text.strip()

    def _call_openai(self, prompt: str) -> str:
        response = _openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.models.temperature,
            max_tokens=cfg.models.max_output_tokens,
        )
        return response.choices[0].message.content.strip()

    def _call_llm(self, prompt: str) -> str:
        if "gemini" in self.model.lower():
            return self._call_gemini(prompt)
        return self._call_openai(prompt)

    def query(
        self,
        question: str,
        verdict_id: str,
        question_id: str = "",
    ) -> QAResult:
        text   = self._load_text(verdict_id)
        prompt = build_prompt(SYSTEM_INSTRUCTION, text, question)

        input_tokens = count_tokens(prompt, self.model)
        cost_usd     = estimate_cost(input_tokens, self.model, cfg.model_pricing)

        t0     = time.perf_counter()
        answer = self._call_llm(prompt)
        latency_s = time.perf_counter() - t0

        log.debug(
            f"[{self.condition_name}] {question_id} | "
            f"tokens={input_tokens:,} cost=${cost_usd:.5f} latency={latency_s:.2f}s"
        )

        return QAResult(
            question_id=question_id,
            question=question,
            verdict_id=verdict_id,
            answer=answer,
            retrieved_chunks=[],
            context_used=text[:500] + "...",  # truncated for logging
            condition=self.condition_name,
            model=self.model,
            input_tokens=input_tokens,
            cost_usd=cost_usd,
            latency_s=latency_s,
        )
