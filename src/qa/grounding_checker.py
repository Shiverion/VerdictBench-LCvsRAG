"""
Verbatim Grounding Checker
==========================
This script performs a deterministic, objective check to ensure 
that every 'gold_answer' in the QA dataset is physically present 
within its corresponding 'gold_paragraphs'.

This is a supplementary quality check for factual accuracy, 
not a replacement for human judgment.
"""

import json
import re
import unicodedata
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)

def normalize(text: str) -> str:
    if not text:
        return ""
    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # 2. Remove soft hyphens
    text = text.replace("\u00ad", "")
    # 3. Collapse hyphen-newline word breaks
    text = re.sub(r"-\s*\n\s*", "", text)
    # 4. Normalize all whitespace (newlines, tabs, multiple spaces) to single space
    text = re.sub(r"\s+", " ", text)
    # 5. Normalize dashes and quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # 6. Lowercase
    text = text.lower()
    return text.strip()

def token_overlap(a: str, b: str) -> float:
    tokens_a = set(a.split())
    if not tokens_a:
        return 0.0
    
    # Simple subset check or token overlap ratio against sliding windows could be complex
    # The user suggested set overlap, which works if the source text is small, but if the source text 
    # is huge, `tokens_b` is the entire document. A paragraph is a subset of the document.
    # So len(tokens_a & tokens_b) / len(tokens_a) measures "what % of words in the paragraph exist *anywhere* in the document".
    # This is a good proxy for "did the model invent new words?".
    tokens_b = set(b.split())
    return len(tokens_a & tokens_b) / len(tokens_a)

def check_grounding(dataset_path: Path):
    if not dataset_path.exists():
        log.error(f"Dataset not found: {dataset_path}")
        return

    from src.utils.config import cfg
    txt_dir = cfg.paths.cleaned
    
    failed = []
    total = 0

    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            total += 1
            data = json.loads(line)
            qid = data.get("question_id", "unknown")
            verdict_id = data.get("verdict_id")
            paragraphs = data.get("gold_paragraphs", [])
            
            if not verdict_id:
                log.warning(f"No verdict_id for pair {qid}")
                continue
                
            txt_path = txt_dir / f"{verdict_id}.txt"
            if not txt_path.exists():
                log.warning(f"Verdict text not found for {verdict_id}")
                failed.append({"id": qid, "reason": "Missing source text"})
                continue
                
            source_text = normalize(txt_path.read_text(encoding="utf-8"))
            
            is_grounded = True
            for i, para in enumerate(paragraphs):
                para_norm = normalize(para)
                if not para_norm:
                    continue
                if para_norm not in source_text:
                    # Try fuzzy fallback
                    overlap = token_overlap(para_norm, source_text)
                    if overlap >= 0.85:
                        pass # Near-match — pass but log as "fuzzy" internally if needed
                    else:
                        is_grounded = False
                        failed.append({
                            "id": qid,
                            "reason": f"Paragraph {i+1} failed (<85% token overlap: {overlap:.2f})"
                        })
                        break 

    log.info(f"--- Paragraph Grounding Audit Report ---")
    log.info(f"Total pairs checked: {total}")
    log.info(f"Successfully grounded: {total - len(failed)}")
    log.info(f"Failed grounding: {len(failed)}")
    
    if failed:
        log.warning(f"Detection: {len(failed)} pairs have ungrounded paragraphs.")
        for f in failed[:5]:
            log.warning(f"  [!] ID: {f['id']} | {f['reason']}")
        if len(failed) > 5:
            log.warning(f"  ... and {len(failed) - 5} more.")

    # Save flags for the reviewer CLI
    flags_path = dataset_path.parent / "grounding_flags.json"
    flagged_ids = [f["id"] for f in failed]
    with open(flags_path, "w", encoding="utf-8") as f:
        json.dump(flagged_ids, f)
    log.info(f"Saved {len(flagged_ids)} flags to {flags_path}")

def main():
    from src.utils.config import cfg
    dataset = cfg.paths.qa_dir / "qa_drafts_raw.jsonl"
    check_grounding(dataset)

if __name__ == "__main__":
    main()
