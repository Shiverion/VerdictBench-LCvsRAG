"""
Query rewriter — rewrites the user question into a retrieval-optimized form.

For multi-hop questions, decomposes into sub-queries that are retrieved
independently, then merged before generation.
"""

from __future__ import annotations

import os

import google.generativeai as genai
from openai import OpenAI

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

_REWRITE_PROMPT = """Anda adalah sistem pengambilan informasi hukum.
Tugas Anda: tulis ulang pertanyaan berikut menjadi 1-3 kueri pencarian yang \
lebih efektif untuk menemukan bagian relevan dari putusan Mahkamah Konstitusi.

Aturan:
- Setiap kueri harus spesifik dan menggunakan kata kunci hukum yang tepat
- Jika pertanyaan membutuhkan informasi dari beberapa bagian dokumen, \
buat kueri terpisah untuk setiap bagian
- Jawab HANYA dengan daftar kueri, satu per baris, tanpa penomoran atau penjelasan

Pertanyaan asli: {question}

Kueri pencarian:"""


def rewrite_query(question: str, model: str | None = None) -> list[str]:
    """
    Rewrite a question into one or more retrieval-optimized queries.

    Args:
        question: Original natural language question.
        model:    LLM to use for rewriting (defaults to phase1 model).

    Returns:
        List of rewritten query strings (1–3 items).
    """
    model = model or cfg.models.phase1_model
    prompt = _REWRITE_PROMPT.format(question=question)

    try:
        if model.startswith("gemini"):
            llm = genai.GenerativeModel(
                model,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=200,
                ),
            )
            raw = llm.generate_content(prompt).text.strip()
        else:
            response = _openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()

        queries = [q.strip() for q in raw.split("\n") if q.strip()]
        # Deduplicate while preserving order; cap at 3
        seen: set[str] = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        result = unique[:3] if unique else [question]

        log.debug(f"Rewritten '{question[:60]}...' → {len(result)} queries")
        return result

    except Exception as e:
        log.warning(f"Query rewrite failed ({e}), falling back to original question.")
        return [question]
