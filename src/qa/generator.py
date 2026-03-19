"""
LLM-assisted QA draft generation (New google-genai SDK).

Uses genai.Client with response_json_schema for strict JSON output.
Ensures 100% syntactical correctness for legal QA pairs.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List

from google import genai
from pydantic import BaseModel, Field
import pandas as pd
from tqdm import tqdm

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

# --- Pydantic Models for Structured Output ---

class QADraftItem(BaseModel):
    question: str = Field(description="pertanyaan dalam Bahasa Indonesia")
    gold_answer: str = Field(description="jawaban berdasarkan teks di atas")
    gold_paragraphs: List[str] = Field(description="paragraf pendukung dari teks")

class QADraftResponse(BaseModel):
    items: List[QADraftItem]

# --- Generation Logic ---

QUESTION_TYPES = {
    "factual_extractive": (
        "Buat 2 pertanyaan faktual yang jawabannya dapat diekstrak langsung dari teks. "
        "Pertanyaan harus spesifik tentang fakta yang tersurat dalam dokumen."
    ),
    "multi_section_reasoning": (
        "Buat 2 pertanyaan yang membutuhkan penalaran lintas beberapa bagian dokumen. "
        "Jawaban tidak bisa ditemukan di satu paragraf saja."
    ),
    "structural": (
        "Buat 1 pertanyaan tentang struktur atau unsur-unsur formal putusan "
        "(panel hakim, tanggal, bukti, saksi, dll)."
    ),
    "boundary": (
        "Buat 1 pertanyaan tentang admissibilitas atau konklusi Mahkamah "
        "(apakah permohonan diterima/ditolak dan atas dasar apa)."
    ),
}

_QA_PROMPT = """Anda adalah asisten pembuatan dataset QA untuk penelitian NLP hukum.

Dokumen putusan MK:
{text_excerpt}

Tugas: {task_instruction}

PENTING:
1. Pastikan semua kutipan teks (gold_paragraphs) akurat.
2. JANGAN menyalin seluruh dokumen. Ambil hanya 1-2 kalimat (maksimal 500 karakter) untuk setiap gold_paragraph.
3. Gunakan Bahasa Indonesia yang formal dan tepat."""


def generate_type_drafts(
    text: str,
    q_type: str,
    instruction: str,
    client: genai.Client,
    model_name: str,
    verdict_id: str,
) -> list[dict]:
    excerpt = text[:10000]
    prompt  = _QA_PROMPT.format(text_excerpt=excerpt, task_instruction=instruction)

    # Retry mechanism: 3 attempts
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": QADraftResponse.model_json_schema(),
                    "temperature": 0.0,
                    "max_output_tokens": 8192,
                },
            )
            
            # Diagnostic: Log finish reason and token usage
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                total_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else "N/A"
                if candidate.finish_reason != "STOP":
                    log.warning(f"Attempt {attempt}: Finished with {candidate.finish_reason} (Tokens: {total_tokens})")
            
            # Use Pydantic to validate and parse
            data = QADraftResponse.model_validate_json(response.text)
            
            items = []
            for it in data.items:
                # Convert Pydantic to dict and add metadata
                d = it.model_dump()
                d["question_id"]   = str(uuid.uuid4())[:8]
                d["verdict_id"]    = verdict_id
                d["question_type"] = q_type
                d["status"]        = "draft"
                items.append(d)
                
            if items:
                return items
                
        except Exception as e:
            log.warning(f"Attempt {attempt} failed for {verdict_id}/{q_type}: {e}")
            if attempt == 3:
                debug_dir = cfg.paths.qa_dir / "debug"
                debug_dir.mkdir(exist_ok=True)
                with open(debug_dir / f"fail_v2_{verdict_id}_{q_type}.txt", "w", encoding="utf-8") as f:
                    f.write(f"ERROR: {e}\n")
    return []


def generate_verdict_drafts(
    verdict_id: str,
    text: str,
    client: genai.Client,
    model_name: str,
) -> list[dict]:
    all_drafts = []
    for q_type, instruction in QUESTION_TYPES.items():
        drafts = generate_type_drafts(text, q_type, instruction, client, model_name, verdict_id)
        all_drafts.extend(drafts)
    return all_drafts


def generate_all_drafts(
    sample_csv: Path,
    cleaned_dir: Path,
    out_path: Path,
    model_name: str | None = None,
) -> int:
    sample = pd.read_csv(sample_csv)
    model_name = model_name or cfg.models.phase1_model
    
    # Initialize new SDK client
    api_key = os.getenv("GOOGLE_API_KEY", "")
    client  = genai.Client(api_key=api_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Generating QA drafts (Strict Mode)"):
            vid      = row["file_id"]
            txt_path = cleaned_dir / f"{vid}.txt"
            if not txt_path.exists():
                continue

            text   = txt_path.read_text(encoding="utf-8")
            drafts = generate_verdict_drafts(vid, text, client, model_name)

            for draft in drafts:
                import json
                f.write(json.dumps(draft, ensure_ascii=False) + "\n")
                total += 1

    log.info(f"Generated {total} QA drafts → {out_path}")
    return total


if __name__ == "__main__":
    generate_all_drafts(
        sample_csv=cfg.paths.sample_50,
        cleaned_dir=cfg.paths.cleaned,
        out_path=cfg.paths.qa_drafts,
        model_name=cfg.models.phase1_model
    )
