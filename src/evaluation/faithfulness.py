"""
Faithfulness evaluation using LLM-as-judge (GPT-4o).

Pipeline:
  1. Decompose the generated answer into atomic factual statements
  2. For each statement, check whether it is supported by the source context
  3. Faithfulness = supported_statements / total_statements

Anti-circularity: judge model (GPT-4o) is always different from the
generation model used in Phase 1 (Gemini 2.5 Flash).
"""

from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ── Prompts ───────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """Anda adalah evaluator sistem QA hukum.

Tugas: Pecah jawaban berikut menjadi pernyataan faktual atomik (satu klaim per baris).
Setiap pernyataan harus bisa dinilai benar/salah secara independen.
Jawab HANYA dengan daftar pernyataan, satu per baris, tanpa penomoran.

Jawaban yang dievaluasi:
{answer}

Pernyataan faktual atomik:"""

_GROUNDING_PROMPT = """Anda adalah evaluator sistem QA hukum.

Tugas: Tentukan apakah pernyataan berikut didukung oleh konteks dokumen yang diberikan.

Konteks dokumen:
{context}

Pernyataan yang dievaluasi:
{statement}

Jawab dengan format JSON berikut SAJA (tanpa markdown):
{{"supported": true/false, "reason": "kalimat singkat penjelasan"}}"""


def _decompose_answer(answer: str, model: str) -> list[str]:
    """Break an answer into atomic factual statements."""
    if not answer.strip():
        return []

    prompt = _DECOMPOSE_PROMPT.format(answer=answer)
    try:
        response = _openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        statements = [s.strip() for s in raw.split("\n") if s.strip()]
        return statements[:20]  # cap to prevent runaway decompositions
    except Exception as e:
        log.warning(f"Decomposition failed: {e}")
        return [answer]   # fallback: treat whole answer as one statement


def _check_grounding(statement: str, context: str, model: str) -> dict:
    """Check if a single statement is supported by the context."""
    prompt = _GROUNDING_PROMPT.format(context=context[:8000], statement=statement)
    try:
        response = _openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        log.warning(f"Grounding check failed: {e}")
        return {"supported": False, "reason": f"evaluation error: {e}"}


def evaluate_faithfulness(
    answer: str,
    context: str,
    judge_model: str | None = None,
) -> dict:
    """
    Compute faithfulness score for a single QA result.

    Args:
        answer:      Generated answer text.
        context:     Source context used for generation (LC text or RAG chunks).
        judge_model: LLM used for evaluation (must differ from generation model).

    Returns:
        Dict with keys:
          - faithfulness      (float 0–1)
          - n_statements      (int)
          - n_supported       (int)
          - statement_details (list of {statement, supported, reason})
    """
    judge_model = judge_model or cfg.models.judge_model

    if not answer.strip():
        return {
            "faithfulness": 0.0,
            "n_statements": 0,
            "n_supported": 0,
            "statement_details": [],
        }

    statements = _decompose_answer(answer, judge_model)
    if not statements:
        return {
            "faithfulness": 0.0,
            "n_statements": 0,
            "n_supported": 0,
            "statement_details": [],
        }

    details = []
    supported_count = 0

    for stmt in statements:
        result = _check_grounding(stmt, context, judge_model)
        if result.get("supported", False):
            supported_count += 1
        details.append({
            "statement": stmt,
            "supported": result.get("supported", False),
            "reason":    result.get("reason", ""),
        })

    n = len(statements)
    faithfulness = supported_count / n if n > 0 else 0.0

    log.debug(
        f"Faithfulness: {supported_count}/{n} statements supported "
        f"→ {faithfulness:.3f}"
    )

    return {
        "faithfulness":      round(faithfulness, 4),
        "n_statements":      n,
        "n_supported":       supported_count,
        "statement_details": details,
    }
