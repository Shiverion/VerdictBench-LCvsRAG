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

from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

class GroundingCheckSchema(BaseModel):
    supported: bool
    reason: str

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))

# ── Prompts ───────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """Anda adalah evaluator sistem QA hukum.

Tugas: Pecah jawaban berikut menjadi pernyataan faktual atomik (satu klaim per baris).
Setiap pernyataan harus bisa dinilai benar/salah secara independen.
KEMBALIKAN HANYA DAFTAR PERNYATAAN, SATU PER BARIS. DILARANG MENYEBUTKAN KATA PENGANTAR ATAU PENOMORAN.

Jawaban yang dievaluasi:
{answer}

Pernyataan faktual atomik:"""

_GROUNDING_PROMPT = """Anda adalah evaluator sistem QA hukum.

Tugas: Tentukan apakah pernyataan berikut didukung oleh konteks dokumen yang diberikan.

Konteks dokumen:
{context}

Pernyataan yang dievaluasi:
{statement}

KEMBALIKAN HANYA OBJEK JSON MURNI TANPA TEKS PENGANTAR APAPUN!!!
DILARANG MEMBERIKAN PENJELASAN EXTRA DILUAR JSON, DILARANG MEMBUKA DENGAN KATA-KATA ("Berikut adalah...", dll).
OUTPUT HARUS 100% RAW JSON: {{"supported": true/false, "reason": "kalimat singkat penjelasan"}}"""

# Clean prompt for Gemini structured output (no JSON format instructions to avoid conflict)
_GROUNDING_PROMPT_STRUCTURED = """Anda adalah evaluator sistem QA hukum.

Tugas: Tentukan apakah pernyataan berikut didukung oleh konteks dokumen yang diberikan.

Konteks dokumen:
{context}

Pernyataan yang dievaluasi:
{statement}

Jawab apakah pernyataan tersebut didukung (supported=true) atau tidak (supported=false) beserta alasannya."""


def _decompose_answer(answer: str, model: str) -> tuple[list[str], int, int]:
    """Break an answer into atomic factual statements."""
    if not answer.strip():
        return [], 0, 0

    prompt = _DECOMPOSE_PROMPT.format(answer=answer)
    try:
        if model.startswith("gemini"):
            response = _gemini.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=400,
                ),
            )
            raw = response.text.strip()
            in_t  = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            out_t = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        else:
            response = _openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()
            in_t  = response.usage.prompt_tokens if response.usage else 0
            out_t = response.usage.completion_tokens if response.usage else 0
        
        statements = [s.strip() for s in raw.split("\n") if s.strip()]
        return statements[:20], in_t, out_t  # cap to prevent runaway decompositions
    except Exception as e:
        log.warning(f"Decomposition failed: {e}")
        return [answer], 0, 0   # fallback: treat whole answer as one statement


def _extract_json_with_flash(raw_text: str) -> dict:
    """Use gemini-2.0-flash with structured output to extract supported/reason from free text."""
    extraction_prompt = (
        f"Extract the grounding verdict from this evaluation text.\n\n"
        f"Evaluation text:\n{raw_text}\n\n"
        f"Determine: is the statement supported (true) or not (false)? "
        f"Extract the reason given."
    )
    response = _gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=extraction_prompt,
        config={
            "temperature": 0.0,
            "max_output_tokens": 150,
            "response_mime_type": "application/json",
            "response_json_schema": GroundingCheckSchema.model_json_schema(),
        },
    )
    parsed = json.loads(response.text.strip())
    sup = parsed.get("supported", False)
    if isinstance(sup, str):
        sup = sup.lower() in ("true", "yes", "1")
    return {"supported": bool(sup), "reason": parsed.get("reason", "")}


def _check_grounding(statement: str, context: str, model: str) -> tuple[dict, int, int]:
    """Check if a single statement is supported by the context.
    
    Two-model pipeline for Gemini:
      1. gemini-3-flash-preview evaluates grounding (free text, stronger reasoning)
      2. gemini-2.0-flash extracts the verdict into structured JSON (reliable output)
    """
    raw = ""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if model.startswith("gemini"):
                prompt = _GROUNDING_PROMPT_STRUCTURED.format(
                    context=context[:8000], statement=statement
                )
                # Step 1: Strong model evaluates grounding (free text)
                response = _gemini.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": 0.0,
                        "max_output_tokens": 300,
                    },
                )
                raw = response.text.strip()
                in_t  = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                out_t = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                
                # Step 2: Try direct JSON parse first (model might return JSON)
                try:
                    clean = raw.replace("```json", "").replace("```", "").strip()
                    start = clean.find("{")
                    end = clean.rfind("}")
                    if start != -1 and end >= start:
                        parsed = json.loads(clean[start:end+1])
                        sup = parsed.get("supported", False)
                        if isinstance(sup, str):
                            sup = sup.lower() in ("true", "yes", "1")
                        return {"supported": bool(sup), "reason": parsed.get("reason", "")}, in_t, out_t
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # Step 3: Use gemini-2.0-flash to extract JSON from free text
                result = _extract_json_with_flash(raw)
                return result, in_t, out_t

            else:
                prompt = _GROUNDING_PROMPT.format(
                    context=context[:8000], statement=statement
                )
                response = _openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                raw = response.choices[0].message.content.strip()
                in_t  = response.usage.prompt_tokens if response.usage else 0
                out_t = response.usage.completion_tokens if response.usage else 0

                # Parse JSON from OpenAI response
                clean = raw.replace("```json", "").replace("```", "").strip()
                start = clean.find("{")
                end = clean.rfind("}")
                if start != -1 and end >= start:
                    clean = clean[start:end+1]
                if not clean or "{" not in clean:
                    raise ValueError(f"Model returned invalid JSON: {raw!r}")
                parsed = json.loads(clean)
                sup = parsed.get("supported", False)
                if isinstance(sup, str):
                    sup = sup.lower() in ("true", "yes", "1")
                return {"supported": bool(sup), "reason": parsed.get("reason", "")}, in_t, out_t

        except Exception as e:
            if attempt < max_retries - 1:
                log.debug(f"Grounding attempt {attempt+1} failed: {e} — retrying")
                continue
            log.warning(f"Grounding check failed: {e}. Raw response: {raw!r}")
            return {"supported": False, "reason": f"evaluation error: {e}"}, 0, 0
    
    return {"supported": False, "reason": "evaluation error: max retries exceeded"}, 0, 0


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
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    statements, dec_in, dec_out = _decompose_answer(answer, judge_model)
    total_in = dec_in
    total_out = dec_out
    
    if not statements:
        return {
            "faithfulness": 0.0,
            "n_statements": 0,
            "n_supported": 0,
            "statement_details": [],
            "usage": {"input_tokens": total_in, "output_tokens": total_out},
        }

    details = []
    supported_count = 0

    for stmt in statements:
        result, gr_in, gr_out = _check_grounding(stmt, context, judge_model)
        total_in += gr_in
        total_out += gr_out
        
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
        "usage":             {"input_tokens": total_in, "output_tokens": total_out},
    }
